// **Real-GPU direct-dispatch checks** for the three stages of states-form attention +
// `state_append` (ADR 0067 decisions 4–7). Wiring into the executor / recipe builder is phase D-3,
// so here the kernels are dispatched directly on hand-built buffers and compared against the CPU
// reference (src/reference/state-attention.ts).
//
// The grid is {full, sliding} × {r=1,2,8} × {P=0 / 1..W−1 / ≥W (across a ring wrap)} ×
// {Q=1, chunk>1} × {with / without pad rows} × {B=1, B>1} × {1 row block / several}.
//
// MUST: do not stop at the comparison. The **structural guarantees** of the states form hide behind
// the tolerance, so these are checked separately:
//   ① the output of an empty row is **exactly 0** (Object.is)
//   ② S outside the predicate is **exactly the −inf bit pattern** (0xff800000)
//   ③ non-resident slot rows and the pad rows of ins hold poison values, so reading them makes the
//      values jump
//   ④ changing the capacity C leaves the output **bit-identical** (the flip side of the work
//      condition = the values do not depend on the capacity)
//   ⑤ splitting into row blocks leaves the output **bit-identical** (ADR 0067 decision 7)
// MUST: show in the same test, by fault injection, that **each check detects its own mutation** (a
// check that detects nothing verifies nothing).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAppend,
  referenceStateAttention,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBase, stateLiveColumns, stateSliding } from "../src/kernels/state-attention.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  assertMutated,
  caseColCap,
  halfScale,
  runStateAppend,
  runStateAttention,
  seeded,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * The comparison tolerance.
 *
 * Rationale: the reference accumulates in f64 and rounds once when storing; the GPU accumulates
 * sequentially in f32, and the difference grows with the dot-product length `D` and the reduction
 * length `live`. Over all 18 `PARITY_CASES` below (`live` from 1 to 4,096) **the measured worst is
 * maxAbs 6.6e-7** (first measured 2026-08-18; the same on the 2026-09-03 re-measurement that added
 * the `live 4096` case — RTX 3080 Ti, Vulkan / outputs of magnitude O(1)), and `atol = 5e-6` adds
 * about 7.6× headroom to it. The worst value comes from the **small** live side; lengthening live
 * actually shrinks the error (main-session measurement (review 2026-09-03) = 4.9e-8 at live 4,096,
 * 4.1e-8 at 16,384, 3.6e-8 at 65,536).
 * MUST: do not use `DEFAULT_TOLERANCE` (atol 1e-5 / rtol 1e-3) — rtol 1e-3 easily passes an error
 * on the scale of "keys outside the window got mixed in" (which is what the checks in this file are
 * for).
 * MUST: `rtol = 0`. The states form legitimately contains **elements whose correct value is exactly
 * 0** (empty rows, contributions outside the predicate), so a relative term does nothing near 0 and
 * only works to hide errors in large elements.
 * NOTE: the measured values depend on the driver's use of fma, so a re-measurement on another
 * adapter must repeat the whole "measure → add headroom" procedure (the test prints the worst value
 * to stdout on every run).
 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/** Deterministic input sequences (no random numbers — a failure would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/**
 * Poison values placed where nothing may read (large enough to exceed the tolerance easily if
 * read).
 */
const SLOT_POISON_K = 9;
const SLOT_POISON_V = 400;

/**
 * Builds the five buffers ①③ read.
 *
 * MUST: put poison values in **non-resident physical slot rows** (evicted rows for sliding /
 * unwritten rows for full) and in **the pad rows of ins** (`row ≥ Q`). By contract neither is read,
 * so an implementation that reads them fails on the values (the only way not to hide a structural
 * error inside the tolerance).
 */
const makeInputs = (spec: StateCase): StateInputs => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = spec;
  const kvPlanes = batch * kvHeads;
  const q = seeded(batch * heads * chunkRows * depth, QUERY);
  const insK = seeded(kvPlanes * chunkRows * depth, KEY);
  const insV = seeded(kvPlanes * chunkRows * depth, VALUE);
  const slotK = seeded(kvPlanes * capacity * depth, (i) => KEY(i + 3));
  const slotV = seeded(kvPlanes * capacity * depth, (i) => VALUE(i + 5));
  // The set of resident physical rows (the image of logical col [colBase, past))
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
    // The pad rows of ins (row ≥ Q) are outside the logical col space [0, P+Q) = never read
    for (let row = query; row < chunkRows; row += 1) {
      for (let d = 0; d < depth; d += 1) {
        insK[(plane * chunkRows + row) * depth + d] = SLOT_POISON_K;
        insV[(plane * chunkRows + row) * depth + d] = SLOT_POISON_V;
      }
    }
  }
  return { q, insK, insV, slotK, slotV };
};

const refInput = (spec: StateCase, inputs: StateInputs): StateAttentionRefInput => ({
  batch: spec.batch,
  heads: spec.heads,
  kvHeads: spec.kvHeads,
  chunkRows: spec.chunkRows,
  depth: spec.depth,
  capacity: spec.capacity,
  window: spec.window,
  past: spec.past,
  query: spec.query,
  scale: halfScale(spec.depth),
  ...inputs,
});

const PARITY_CASES: readonly StateCase[] = [
  // ── full ──────────────────────────────────────────────────────────────────
  // The smallest decode shape (Q=1, P=0 = a causal triangle with only the self-reference)
  {
    name: "full r1 decode P0",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 16,
    window: 0,
    past: 0,
    query: 1,
  },
  // A prefill chunk (Q=M, no pad rows) × B>1
  {
    name: "full r1 prefill B2",
    batch: 2,
    heads: 4,
    kvHeads: 4,
    chunkRows: 6,
    depth: 8,
    capacity: 32,
    window: 0,
    past: 0,
    query: 6,
  },
  // With pad rows (Q<M) × GQA r=2 × an uneven D
  {
    name: "full r2 pad B2 D6",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    chunkRows: 6,
    depth: 6,
    capacity: 32,
    window: 0,
    past: 5,
    query: 3,
  },
  // MQA 8:1 decode (Gemma 4 E2B type) × a larger P
  {
    name: "full r8 MQA decode",
    batch: 2,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 17,
    query: 1,
  },
  // Three row blocks (including a partial block)
  {
    name: "full r1 rowblock 4x3",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 9,
    depth: 8,
    capacity: 32,
    window: 0,
    past: 3,
    query: 9,
    rowsBlock: 4,
  },
  // Tile boundaries (live exactly a multiple of 16, D exactly a multiple of 16)
  {
    name: "full r2 tile exact",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 32,
    capacity: 64,
    window: 0,
    past: 16,
    query: 16,
  },
  // ── sliding ───────────────────────────────────────────────────────────────
  // P=0 (a decode where the window has no effect)
  {
    name: "sliding W8 P0 decode",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 8,
    window: 8,
    past: 0,
    query: 1,
  },
  // 0 < P < W−1 (the whole past is resident, no wrap) + pad rows
  {
    name: "sliding W8 P3 pad",
    batch: 2,
    heads: 4,
    kvHeads: 4,
    chunkRows: 4,
    depth: 8,
    capacity: 8,
    window: 8,
    past: 3,
    query: 2,
  },
  // P = W−1 (the boundary where the lower bound starts to apply)
  {
    name: "sliding W4 P3 edge",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 3,
    depth: 4,
    capacity: 4,
    window: 4,
    past: 3,
    query: 2,
  },
  // P ≫ W (after the ring has wrapped many times — the physical-row mapping matters)
  {
    name: "sliding W4 P10 wrap",
    batch: 2,
    heads: 4,
    kvHeads: 4,
    chunkRows: 4,
    depth: 8,
    capacity: 4,
    window: 4,
    past: 10,
    query: 2,
  },
  // C > W (the ring's modulus is the capacity — fails if either read or write turns into `% W`)
  {
    name: "sliding W4 C8 wrap",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 3,
    depth: 4,
    capacity: 8,
    window: 4,
    past: 9,
    query: 2,
  },
  // GQA r=2 × wrap × pad rows
  {
    name: "sliding W6 r2 wrap pad",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    chunkRows: 5,
    depth: 6,
    capacity: 6,
    window: 6,
    past: 13,
    query: 3,
  },
  // MQA 8:1 × wrap decode (the Gemma 4 E2B sliding-layer type)
  {
    name: "sliding W8 r8 MQA",
    batch: 2,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 21,
    query: 1,
  },
  // **A shape where empty rows occur legitimately** (pad rows drop out of the window)
  {
    name: "sliding W2 empty rows",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 4,
    depth: 4,
    capacity: 2,
    window: 2,
    past: 0,
    query: 1,
  },
  // Q > W (this step has more keys than the window — legitimate, since current is read from ins)
  {
    name: "sliding W2 Q4 gt W",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 4,
    depth: 4,
    capacity: 2,
    window: 2,
    past: 5,
    query: 4,
  },
  // Two row blocks × sliding
  {
    name: "sliding W6 rowblock",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 8,
    depth: 8,
    capacity: 6,
    window: 6,
    past: 20,
    query: 8,
    rowsBlock: 5,
  },
  // **P near the u32 maximum** (the only shape where the direction of the lower-bound predicate's
  // subtraction changes the values). With the additive form `col + W > limit`, at the causal
  // diagonal `col = P` for `P + 0`, `P + 2` crosses 2^32 and wraps to 0, the self-referencing key
  // is judged out of window, and the values jump.
  {
    name: "sliding W2 P near u32 max",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 2,
    window: 2,
    past: 0xfffffffe,
    query: 1,
  },
  // ── Coverage statement ────────────────────────────────────────────────────────
  // A decode with the reduction length `live` extended to production scale (four digits). The grid
  // above has live ≤ a few dozen, so this checks that "the rationale for `atol` is not an
  // extrapolation from small live" (review W-G1-1).
  // The shape is the Gemma 4 E2B decode (MQA 8:1, D 32). **Append at the end** — moving the indices
  // above would shift the references of `PREDICATE_CASES` / `INJECTIONS`.
  {
    name: "full r8 MQA decode live 4096",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 32,
    capacity: 4096,
    window: 0,
    past: 4095,
    query: 1,
  },
];

/** The rows the reference judges "empty" (flat indices into `[B·H, M]`). */
const emptyRows = (spec: StateCase): readonly number[] => {
  const rows: number[] = [];
  const base = stateColumnBase(spec.window, spec.past);
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
    for (let row = 0; row < spec.chunkRows; row += 1) {
      const limit = spec.past + row;
      let any = false;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        if (col <= limit && (!stateSliding(spec.window) || col + spec.window > limit)) {
          any = true;
          break;
        }
      }
      if (!any) rows.push(plane * spec.chunkRows + row);
    }
  }
  return rows;
};

/** Flat indices into `[B·H, M]` of the pad rows (`row ≥ Q`). */
const padRows = (spec: StateCase): readonly number[] => {
  const rows: number[] = [];
  for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
    for (let row = spec.query; row < spec.chunkRows; row += 1) {
      rows.push(plane * spec.chunkRows + row);
    }
  }
  return rows;
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

test.skipIf(!GPU_AVAILABLE)(
  "states-form attention matches the CPU reference (real GPU, full / sliding × GQA × row blocks)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    let worstAbs = 0;
    let worstRel = 0;
    try {
      for (const spec of PARITY_CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const actual = await runStateAttention(gpu.device, spec, inputs, { cache });
        const report = compareTensors(
          { dtype: "f32", data: actual.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(report.pass, true, `${spec.name}: ${formatAllclose(report)}`);
        worstAbs = Math.max(worstAbs, report.maxAbsError);
        worstRel = Math.max(worstRel, report.maxRelError);
        // ① pad rows are **exactly 0** (pins separately the errors that hide inside the
        //   tolerance). Empty rows are a subset of them (a valid row is never empty thanks to the
        //   causal self-reference), so check the inclusion too — if it broke, the ground for the
        //   pad rows' zero write covering "empty row → 0" would be gone
        const empties = emptyRows(spec);
        const pads = padRows(spec);
        assertEquals(
          empties.filter((row) => !pads.includes(row)),
          [],
          `${spec.name}: an empty row appeared outside the pad rows (empty rows ⊂ pad rows is broken)`,
        );
        for (const row of pads) {
          for (let d = 0; d < spec.depth; d += 1) {
            assertEquals(
              Object.is(actual.out[row * spec.depth + d], 0),
              true,
              `${spec.name}: the output of pad row ${row} is not exactly 0 (${
                actual.out[row * spec.depth + d]
              })`,
            );
          }
        }
        // ③ no poison value remains in the output (all M rows including the pad rows were written =
        // full-write)
        assertEquals(
          actual.out.some((value) => value === STATE_S_POISON),
          false,
          `${spec.name}: poison remains in the output (③PV did not write every row)`,
        );
      }
    } finally {
      gpu.destroy();
    }
    // Print the measured worst value as the tolerance's rationale (loosening the tolerance or a
    // growing error shows up in the order of magnitude at once)
    console.log(`[states parity] maxAbs=${worstAbs} maxRel=${worstRel}`);
  },
);

/**
 * Single-block cases for checking the predicate directly (S is read back and inspected as bit
 * patterns).
 *
 * Three conditions: ① valid rows have **both** columns inside and outside the predicate (the
 * causal upper bound in every case; the sliding lower bound is exercised by `P10 wrap` /
 * `W6 r2 wrap pad`) ② pad rows exist (the check that S is not written) ③ one row block (local row =
 * global row).
 * NOTE: since phase D-7, ① covers only the valid rows, so `sliding W2 empty rows` (with Q=1 the
 * only valid row is row 0, and its one column is always inside the predicate) cannot produce a
 * column outside the predicate here. That case is still exercised by the parity side and the fault
 * injection (③PV's zero write of pad rows).
 */
const PREDICATE_CASES: readonly StateCase[] = [
  PARITY_CASES[2],
  PARITY_CASES[7],
  PARITY_CASES[9],
  PARITY_CASES[11],
];

test.skipIf(!GPU_AVAILABLE)(
  "the S that ①QK writes is exactly −inf outside the predicate, and every column in the live range is written (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      for (const spec of PREDICATE_CASES) {
        const inputs = makeInputs(spec);
        const { scores } = await runStateAttention(gpu.device, spec, inputs, { cache });
        const colCap = caseColCap(spec);
        const base = stateColumnBase(spec.window, spec.past);
        const live = stateLiveColumns(spec.window, spec.past, spec.query);
        const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
        // Only single-block cases are listed, so rowOffset = 0 = the valid rows are the first Q
        // rows
        const effRows = Math.min(spec.query, rowsBlock);
        const bits = bitsOf(scores);
        let masked = 0;
        let written = 0;
        let skipped = 0;
        for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
          for (let local = 0; local < effRows; local += 1) {
            // Only single-block cases are listed, so local row = global row
            const limit = spec.past + local;
            for (let cl = 0; cl < live; cl += 1) {
              const col = base + cl;
              const inWindow =
                col <= limit && (!stateSliding(spec.window) || limit - col < spec.window);
              const at = (plane * rowsBlock + local) * colCap + cl;
              if (inWindow) {
                assertEquals(
                  Number.isFinite(scores[at]),
                  true,
                  `${spec.name}: (${local},${cl}) inside the predicate is not finite`,
                );
                assertEquals(
                  scores[at] === STATE_S_POISON,
                  false,
                  `${spec.name}: poison remains at (${local},${cl}) inside the predicate (not written)`,
                );
                written += 1;
              } else {
                assertEquals(
                  bits[at],
                  0xff800000,
                  `${spec.name}: (${local},${cl}) outside the predicate is not the −inf bit pattern`,
                );
                masked += 1;
              }
            }
          }
          // **Not one word** of S in the pad rows is written (direct evidence that the work scales
          // with Q, not M — the poison stays as it is. If this were written, ① would be looping
          // over all M rows)
          for (let local = effRows; local < rowsBlock; local += 1) {
            for (let cl = 0; cl < live; cl += 1) {
              const at = (plane * rowsBlock + local) * colCap + cl;
              assertEquals(
                scores[at],
                STATE_S_POISON,
                `${spec.name}: S of pad row (${local},${cl}) was written (① looped over M rows)`,
              );
              skipped += 1;
            }
          }
        }
        // The checks do fire (the shapes are chosen so that columns on both sides and pad rows
        // exist)
        assert(written > 0, `${spec.name}: no column inside the predicate`);
        assert(
          masked > 0,
          `${spec.name}: no column outside the predicate (a shape where the lower / upper bound has no effect)`,
        );
        assert(skipped > 0, `${spec.name}: no pad row (the work check never fires)`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "①③ read back what state_append wrote into the slots (ring round trip, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // W=4 / C=8 (the ring's modulus is the capacity — fails if either read or write turns into
      // `% W`), P=6: the shape after the ring has wrapped once
      const spec: StateCase = {
        name: "append→attention round trip",
        batch: 2,
        heads: 4,
        kvHeads: 2,
        chunkRows: 3,
        depth: 4,
        capacity: 8,
        window: 4,
        past: 6,
        query: 2,
      };
      const kvPlanes = spec.batch * spec.kvHeads;
      const appendSpec = {
        kvPlanes,
        chunkRows: spec.chunkRows,
        depth: spec.depth,
        capacity: spec.capacity,
        window: spec.window,
      };
      // Build real slots by stacking "the appends of the previous steps" three times
      // (P: 0→2→4→6)
      let slotK = seeded(kvPlanes * spec.capacity * spec.depth, () => STATE_S_POISON);
      let slotV = slotK.slice();
      let expectedK = slotK.slice();
      let expectedV = slotV.slice();
      for (let step = 0; step < 3; step += 1) {
        const past = step * 2;
        const xK = seeded(kvPlanes * spec.chunkRows * spec.depth, (i) => KEY(i + step * 13));
        const xV = seeded(kvPlanes * spec.chunkRows * spec.depth, (i) => VALUE(i + step * 7));
        const lengths = { ...appendSpec, past, query: 2 };
        slotK = await runStateAppend(gpu.device, lengths, xK, slotK, { cache });
        slotV = await runStateAppend(gpu.device, lengths, xV, slotV, { cache });
        // MUST: a copy, so **bit-identical** (the reference is a copy of the slot written with the
        // same mapping)
        expectedK = referenceStateAppend({ ...lengths, x: xK, slot: expectedK })
          .data as Float32Array<ArrayBuffer>;
        expectedV = referenceStateAppend({ ...lengths, x: xV, slot: expectedV })
          .data as Float32Array<ArrayBuffer>;
        assertEquals([...bitsOf(slotK)], [...bitsOf(expectedK)], `step ${step}: k slot`);
        assertEquals([...bitsOf(slotV)], [...bitsOf(expectedV)], `step ${step}: v slot`);
      }
      // Confirm that the rows overwritten by the wrap (logical col 0,1 = physical rows 0,1, already
      // overwritten by logical col 4,5) really are "not the old values"
      assert(
        slotK.some((value) => value !== STATE_S_POISON),
        "append wrote no row at all",
      );
      // Read back: feed these slots to ①③ as they are (the pad rows of ins keep their poison)
      const inputs = { ...makeInputs(spec), slotK, slotV };
      const expected = referenceStateAttention(refInput(spec, inputs));
      const actual = await runStateAttention(gpu.device, spec, inputs, { cache });
      const report = compareTensors({ dtype: "f32", data: actual.out }, expected, STATE_TOLERANCE);
      assertEquals(report.pass, true, `ring round trip: ${formatAllclose(report)}`);
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **Deduplication** when a chunk with `Q > C` is written into a slot with `C > W` (the ring wraps
 * within a single append).
 *
 * The ring's modulus moved from the window to the capacity, so the deduplication condition also
 * changed from `row + W ≥ Q` to `row + C ≥ Q`. Cutting at the window would leave the rows
 * `W ≤ row < C` **unwritten even though they alias nothing** (no exception, no warning; only those
 * physical rows keep the leftovers of the previous step).
 *
 * MUST: build the expected values **directly from the logical rows**, not from the reference
 * implementation (the reference shares the same mapping, so if both sides had the same error the
 * comparison would be a tautology). The contents of physical row `r` are "the last of the logical
 * rows that map to `r`" = the largest `col` below `Q` with `col ≡ r (mod C)`.
 */
test.skipIf(!GPU_AVAILABLE)(
  "an append with Q > C keeps only the last logical row per physical row (deduplication with C > W, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // A sliding slot with 8 rows of headroom (the draft limit of speculative decoding).
      const spec = {
        kvPlanes: 2,
        chunkRows: 20,
        depth: 4,
        capacity: 16,
        window: 8,
        past: 0,
        query: 20,
      };
      const x = seeded(spec.kvPlanes * spec.chunkRows * spec.depth, KEY);
      const slot = seeded(spec.kvPlanes * spec.capacity * spec.depth, () => STATE_S_POISON);
      const written = await runStateAppend(gpu.device, spec, x, slot, { cache });

      for (let plane = 0; plane < spec.kvPlanes; plane += 1) {
        for (let row = 0; row < spec.capacity; row += 1) {
          let last = -1;
          for (let col = row; col < spec.query; col += spec.capacity) last = col;
          assert(last >= 0, `no logical row maps to physical row ${row} (the grid is built wrong)`);
          for (let d = 0; d < spec.depth; d += 1) {
            assertEquals(
              written[(plane * spec.capacity + row) * spec.depth + d],
              x[(plane * spec.chunkRows + last) * spec.depth + d],
              `plane ${plane} row ${row}: logical row ${last} did not survive`,
            );
          }
        }
      }

      // Fault injection: cutting the deduplication at the window leaves the logical rows
      // `8 ≤ row < 16` unwritten, and poison remains.
      const mutated = await runStateAppend(gpu.device, spec, x, slot, {
        cache,
        mutate: replaceIn(
          ["append"],
          "row + params.capacity >= query",
          "row + params.window >= query",
        ),
      });
      assertEquals(
        [...bitsOf(mutated)].every((value, at) => value === bitsOf(written)[at]),
        false,
        "fault injection 'cut the deduplication at the window' was not detected",
      );
      assert(
        mutated.some((value) => value === STATE_S_POISON),
        "every row is filled even with the window-cut mutation (the grid does not satisfy Q > C)",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "two dispatches on the same input are bit-identical (determinism of the states form, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      for (const spec of [PARITY_CASES[5], PARITY_CASES[11]]) {
        const inputs = makeInputs(spec);
        const first = await runStateAttention(gpu.device, spec, inputs, { cache });
        const second = await runStateAttention(gpu.device, spec, inputs, { cache });
        assertEquals(
          [...bitsOf(first.out)],
          [...bitsOf(second.out)],
          `${spec.name}: the second run differs`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the capacity C and the row-block split do not move the output bits (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // ④ capacity independence: with the same (P, Q), growing C from 32 → 4096 leaves the output
      // bit-identical (only S's column stride and the unread poison area grow = the flip side of
      // the work condition)
      const small: StateCase = {
        name: "capacity 32",
        batch: 2,
        heads: 4,
        kvHeads: 2,
        chunkRows: 4,
        depth: 8,
        capacity: 32,
        window: 0,
        past: 7,
        query: 3,
      };
      const large: StateCase = { ...small, name: "capacity 4096", capacity: 4096 };
      const smallInputs = makeInputs(small);
      // MUST: the large-capacity slots get **the same values in the same physical rows** (the
      // generator depends on the index, so regenerating at another capacity would shift each
      // plane's origin and compare "two runs with different inputs")
      const relayout = (source: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
        const planes = small.batch * small.kvHeads;
        const grown = seeded(planes * large.capacity * small.depth, () => SLOT_POISON_V);
        for (let plane = 0; plane < planes; plane += 1) {
          const from = plane * small.capacity * small.depth;
          grown.set(
            source.subarray(from, from + small.capacity * small.depth),
            plane * large.capacity * small.depth,
          );
        }
        return grown;
      };
      const largeInputs: StateInputs = {
        ...smallInputs,
        slotK: relayout(smallInputs.slotK),
        slotV: relayout(smallInputs.slotV),
      };
      const smallOut = await runStateAttention(gpu.device, small, smallInputs, { cache });
      const largeOut = await runStateAttention(gpu.device, large, largeInputs, { cache });
      assertEquals(
        [...bitsOf(smallOut.out)],
        [...bitsOf(largeOut.out)],
        "the output moved with the capacity",
      );

      // ⑤ row-block independence: 1 block and 3 blocks (with a partial one) give bit-identical
      // output (ADR 0067 decision 7)
      const whole: StateCase = {
        name: "rowblock whole",
        batch: 1,
        heads: 4,
        kvHeads: 1,
        chunkRows: 7,
        depth: 8,
        capacity: 6,
        window: 6,
        past: 11,
        query: 5,
      };
      const split: StateCase = { ...whole, name: "rowblock 3", rowsBlock: 3 };
      const inputs = makeInputs(whole);
      const wholeOut = await runStateAttention(gpu.device, whole, inputs, { cache });
      const splitOut = await runStateAttention(gpu.device, split, inputs, { cache });
      assertEquals(
        [...bitsOf(wholeOut.out)],
        [...bitsOf(splitOut.out)],
        "the output moved with the row blocks",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/** One fault injection (`apply` rewrites only the target kernel's WGSL). */
type Injection = {
  readonly label: string;
  /** Which case it runs on (a shape where the mutation shows in the values). */
  readonly spec: StateCase;
  readonly apply: (kernel: string, wgsl: string) => string;
};

const replaceIn =
  (
    target: readonly string[],
    from: string,
    to: string,
  ): ((kernel: string, wgsl: string) => string) =>
  (kernel, wgsl) => {
    if (!target.includes(kernel)) return wgsl;
    const mutated = wgsl.replaceAll(from, to);
    assertMutated(wgsl, mutated, `${kernel}: ${from}`);
    return mutated;
  };

const INJECTIONS: readonly Injection[] = [
  // ① Drop the sliding lower-bound predicate (= an implementation with only the causal upper
  // bound; the high-severity finding of review round 5 on ADR 0067 decision 4)
  {
    label: "remove the sliding lower-bound predicate",
    spec: PARITY_CASES[9],
    apply: replaceIn(["qk"], " && (limit - col) < params.window", ""),
  },
  // ①' Turn the lower bound's subtraction back into the additive form that wraps in u32 (the
  // values change only with P near the u32 maximum)
  {
    label: "replace the lower-bound predicate with the additive form col + W > limit",
    spec: PARITY_CASES[16],
    apply: replaceIn(["qk"], "(limit - col) < params.window", "col + params.window > limit"),
  },
  // ② Replace only the read-side ring mapping with `% W` (breaks the MUST that read and write use
  //   the same expression. Now that the modulus is the capacity rather than the window, going
  //   back to the window is what breaks the rule — it shows in the values only with `C > W`)
  {
    label: "replace the read-side ring mapping with col % window",
    spec: PARITY_CASES[10],
    apply: replaceIn(["qk", "pv"], "col % params.capacity", "col % params.window"),
  },
  // ③ Remove ③PV's zero write of pad rows (pad rows fall into the live scan and consume S / stats
  //   that ①② do not cover).
  //   NOTE: since phase D-7, ①② cover only the valid rows, so **empty rows no longer occur
  //   structurally** (a valid row is never empty thanks to the causal self-reference; empty rows ⊂
  //   pad rows). ②'s empty-row guard stays as a defense but can no longer have a detector — this
  //   branch is what guards "empty row / pad row → exactly 0" (ADR 0067 decision 6 / ADR 0066
  //   addendum 6), so the fault injection moved its target there.
  {
    label: "remove ③PV's zero write of pad rows",
    spec: PARITY_CASES[13],
    apply: replaceIn(["pv"], "    out[at] = 0.0;\n    return;\n", ""),
  },
  // ④ Ignore r in the GQA kv mapping (the head correspondence breaks)
  {
    label: "ignore r in the kv-plane mapping",
    spec: PARITY_CASES[11],
    apply: replaceIn(["qk", "pv"], "z / params.kv_repeat", "z"),
  },
  // ⑤ ①QK does not write outside the predicate (② consumes S's leftovers — the reason the whole
  // live range must be written)
  {
    label: "do not write S outside the predicate",
    spec: PARITY_CASES[9],
    apply: replaceIn(
      ["qk"],
      "  s[(z * params.rows_block",
      "  if (!in_window(col, past + row)) { return; }\n  s[(z * params.rows_block",
    ),
  },
  // ⑥ Apply the half scale on one side only (the values are off by a factor of √ — a mistake in
  // copying the contract)
  {
    label: "apply the half scale on the k side only",
    spec: PARITY_CASES[2],
    apply: replaceIn(["qk"], "(q[q_base + d] * params.scale)", "q[q_base + d]"),
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "fault injection: every mutated version turns the comparison red (proof that the checks really detect, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      for (const injection of INJECTIONS) {
        const spec = injection.spec;
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const actual = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          mutate: injection.apply,
        });
        const report = compareTensors(
          { dtype: "f32", data: actual.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(
          report.pass,
          false,
          `fault injection '${injection.label}' was not detected on ${spec.name}`,
        );
      }
      // ⑦ Drop the ring mapping of `state_append` (the write side — the round trip's bit-identity
      //    check is the detector).
      //    MUST: make `P` cross the **capacity** (`P ≥ C`). Now that the modulus is the capacity,
      //    with `P < C` the mapping is indistinguishable from the identity, the mutation never
      //    shows in the values and the check stays green.
      const appendSpec = {
        kvPlanes: 2,
        chunkRows: 3,
        depth: 4,
        capacity: 8,
        window: 4,
        past: 10,
        query: 2,
      };
      const x = seeded(appendSpec.kvPlanes * appendSpec.chunkRows * appendSpec.depth, KEY);
      const slot = seeded(appendSpec.kvPlanes * appendSpec.capacity * appendSpec.depth, VALUE);
      const mutated = await runStateAppend(gpu.device, appendSpec, x, slot, {
        cache,
        mutate: replaceIn(["append"], "col % params.capacity", "col"),
      });
      const expected = referenceStateAppend({ ...appendSpec, x, slot })
        .data as Float32Array<ArrayBuffer>;
      assertEquals(
        [...bitsOf(mutated)].every((value, at) => value === bitsOf(expected)[at]),
        false,
        "fault injection 'drop the append ring mapping' was not detected",
      );
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Propagation of non-finite values (ADR 0020's NaN propagation contract holds for the states form
// too)
// ---------------------------------------------------------------------------

/** The flat index into `[B·H, M, D]` (shared by q and out). */
const rowBase = (spec: StateCase, plane: number, row: number): number =>
  (plane * spec.chunkRows + row) * spec.depth;

/**
 * Sets **this step's first row** of ins (`[B·Hkv, M, D]`) to `value` on every kv plane. It goes on
 * every plane because under GQA the output plane `z` reads only kv plane `z / r` (poisoning only
 * one would claim "should become non-finite" even for planes that were not poisoned).
 */
const poisonFirstInsRow = (
  spec: StateCase,
  ins: Float32Array<ArrayBuffer>,
  value: number,
): void => {
  for (let plane = 0; plane < spec.batch * spec.kvHeads; plane += 1) {
    const base = plane * spec.chunkRows * spec.depth;
    for (let d = 0; d < spec.depth; d += 1) ins[base + d] = value;
  }
};

/**
 * **The only real-GPU check** that ② uses `nan_max` (a bit-pattern NaN test — ADR 0020) for the
 * row max.
 *
 * Plain `max` drops NaN at the spec level ("e2 if e1 < e2, otherwise e1"), so `amax` of an all-NaN
 * row stays at the identity −inf, ② **judges it empty and writes stats (0,0)**, and ③ outputs
 * exactly 0. There is neither an exception nor a NaN, so the whole reason the key
 * `attention_state_stats:v2` exists (the MUST in src/kernels/state-attention.ts) would be lost
 * silently.
 *
 * MUST: do not let this become a tautology. Also check that the same row is a **non-zero finite
 * value** in a control run without NaN (closes off passing because "that row was 0 anyway").
 */
test.skipIf(!GPU_AVAILABLE)(
  "the states form keeps an all-NaN q row NaN in the output without leaking into other rows (②'s nan_max, real GPU)",
  async () => {
    const spec = PARITY_CASES[1]; // full r1 prefill B2 (B2 H4 M6 D8 C32 P0 Q6, no pad rows)
    const plane = 1;
    const row = 2;
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // Control (no NaN) — the check presupposes that the same row is a non-zero finite value
      const clean = makeInputs(spec);
      const control = await runStateAttention(gpu.device, spec, clean, { cache });
      const at = rowBase(spec, plane, row);
      assert(
        [...control.out.slice(at, at + spec.depth)].every(Number.isFinite),
        "the target row of the control run is not finite",
      );
      assert(
        [...control.out.slice(at, at + spec.depth)].some((value) => value !== 0),
        "the target row of the control run is all 0 (the NaN check would be a tautology)",
      );

      for (const pvReduce of ["sequential", "parallel"] as const) {
        const inputs = makeInputs(spec);
        for (let d = 0; d < spec.depth; d += 1) inputs.q[at + d] = Number.NaN;
        const actual = await runStateAttention(gpu.device, spec, inputs, { cache, pvReduce });
        const where = `③ ${pvReduce}`;
        // ① every D component of the (z, row) given NaN is NaN
        for (let d = 0; d < spec.depth; d += 1) {
          assert(
            Number.isNaN(actual.out[at + d]),
            `${where}: output [${d}] of the NaN row is ${actual.out[at + d]} (NaN was not preserved)`,
          );
        }
        // ③ it has not turned into exactly 0 (going back to plain max makes this 0)
        for (let d = 0; d < spec.depth; d += 1) {
          assertEquals(
            Object.is(actual.out[at + d], 0),
            false,
            `${where}: output [${d}] of the NaN row is exactly 0 (it turned into an empty-row verdict)`,
          );
        }
        // ② other rows and other z stay finite (the states form keeps independent statistics per
        // row)
        for (let p = 0; p < spec.batch * spec.heads; p += 1) {
          for (let r = 0; r < spec.chunkRows; r += 1) {
            if (p === plane && r === row) continue;
            const other = rowBase(spec, p, r);
            for (let d = 0; d < spec.depth; d += 1) {
              assert(
                Number.isFinite(actual.out[other + d]),
                `${where}: NaN leaked into the output of plane ${p} row ${r}`,
              );
            }
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * ③PV's zero write of pad rows **also closes the 0·NaN hole** (the branch in
 * src/kernels/state-attention.ts that pins "values meaningless by contract" to 0). Even with
 * non-finite V mixed in, pad rows never scan live, so they stay exactly 0 — an implementation that
 * scans first and multiplies by 0 afterwards gets `0 · ±Inf = NaN`.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the states form propagates non-finite V only to valid rows, and pad rows / empty rows stay exactly 0 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      // ② +Infinity (a shape with pad rows) — make **this step's first row** of ins non-finite on
      // every kv plane
      const padSpec = PARITY_CASES[2]; // full r2 pad B2 D6 (M6 Q3 P5)
      const padInputs = makeInputs(padSpec);
      poisonFirstInsRow(padSpec, padInputs.insV, Number.POSITIVE_INFINITY);
      const padOut = await runStateAttention(gpu.device, padSpec, padInputs, { cache });
      for (let plane = 0; plane < padSpec.batch * padSpec.heads; plane += 1) {
        for (let r = 0; r < padSpec.query; r += 1) {
          const at = rowBase(padSpec, plane, r);
          assert(
            [...padOut.out.slice(at, at + padSpec.depth)].some((value) => !Number.isFinite(value)),
            `+Inf: plane ${plane} valid row ${r} stays finite (V was not read)`,
          );
        }
      }
      for (const row of padRows(padSpec)) {
        for (let d = 0; d < padSpec.depth; d += 1) {
          assertEquals(
            Object.is(padOut.out[row * padSpec.depth + d], 0),
            true,
            `+Inf: the output of pad row ${row} is not exactly 0 (the 0·Inf hole)`,
          );
        }
      }

      // ③ −Infinity (a shape where empty rows occur legitimately) — empty rows ⊂ pad rows, so the
      // same branch guards them
      const emptySpec = PARITY_CASES[13]; // sliding W2 empty rows (M4 Q1)
      const emptyInputs = makeInputs(emptySpec);
      poisonFirstInsRow(emptySpec, emptyInputs.insV, Number.NEGATIVE_INFINITY);
      const emptyOut = await runStateAttention(gpu.device, emptySpec, emptyInputs, { cache });
      assert(emptyRows(emptySpec).length > 0, "no empty row at all (the check never fires)");
      for (const row of emptyRows(emptySpec)) {
        for (let d = 0; d < emptySpec.depth; d += 1) {
          assertEquals(
            Object.is(emptyOut.out[row * emptySpec.depth + d], 0),
            true,
            `−Inf: the output of empty row ${row} is not exactly 0`,
          );
        }
      }
      for (let plane = 0; plane < emptySpec.batch * emptySpec.heads; plane += 1) {
        const at = rowBase(emptySpec, plane, 0);
        assert(
          [...emptyOut.out.slice(at, at + emptySpec.depth)].some(
            (value) => !Number.isFinite(value),
          ),
          `−Inf: the valid row of plane ${plane} stays finite (V was not read)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The detector for ②'s **empty-row guard** (a row with `amax == -inf` never runs the Σ loop +
 * writes stats (0,0)). Since phase D-7, ①② cover only the valid rows, so empty rows do not occur
 * structurally — so a **fault injection that forces ①QK's predicate to false** creates the state
 * "every valid row is empty" and hits the guard directly.
 *
 * Without the guard, `exp(-inf - (-inf))` = `exp(NaN)` = NaN enters the denominator and the whole
 * row becomes NaN.
 */
test.skipIf(!GPU_AVAILABLE)(
  "even with ①QK's predicate forced to false, ②'s empty-row guard keeps the output exactly 0 (no NaN, real GPU)",
  async () => {
    const spec = PARITY_CASES[2]; // full r2 pad B2 D6 (P5 Q3 M6)
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      const inputs = makeInputs(spec);
      const injected = await runStateAttention(gpu.device, spec, inputs, {
        cache,
        mutate: replaceIn(["qk"], "if (in_window(col, past + row)) {", "if (false) {"),
      });
      // ① every element is exactly 0 (empty row → (0,0) → exp(-inf - 0) · 0 = 0)
      const nonZero = [...injected.out].findIndex((value) => !Object.is(value, 0));
      assertEquals(
        nonZero,
        -1,
        `empty-row guard: index ${nonZero} is not exactly 0 (${injected.out[nonZero] ?? ""})`,
      );
      // ② not a single NaN (direct evidence that exp(-inf - (-inf)) is avoided)
      assertEquals(
        injected.out.some(Number.isNaN),
        false,
        "empty-row guard: NaN remains in the output",
      );
      // ③ no poison remains either (③PV wrote every row = full-write holds for empty rows too)
      assertEquals(
        injected.out.some((value) => value === STATE_S_POISON),
        false,
        "empty-row guard: poison remains in the output",
      );
      // Control: without the injection the output is non-zero (evidence that this injection is not
      // a "make everything 0" kind)
      const control = await runStateAttention(gpu.device, spec, inputs, { cache });
      assert(
        [...control.out].some((value) => value !== 0),
        "the control run's output is all 0 (the injection's effect cannot be seen)",
      );
    } finally {
      gpu.destroy();
    }
  },
);
