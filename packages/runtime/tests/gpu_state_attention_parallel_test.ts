// A/B band checks (ADR 0058 decision 4 ②) for the **two parallel-reduction stages** of states-form
// attention (one preset — `stateAttentionReduce: "parallel"`):
//   ③' = KV-parallel reduction of ③PV (perf-ledger K-12)
//   ①' = D-parallel reduction of ①QK (perf-ledger K-14)
// The checks of the reference path (sequential ① ③) stay in gpu_state_attention_test.ts without a
// single character changed (the freeze of decision 4 ① of the same ADR).
//
// Both change the reduction order, so neither is bit-identical to the reference path. Checked here:
//   ① allclose against the f64 reference (the same tolerance as the reference path — changing the
//      order does not change the band of correctness)
//   ② the A/B band against the reference path (measured drift × margin — the derivation tables of
//      `AB_TOLERANCE` / `QK_AB_TOLERANCE` below)
//   ③ the same structural guarantees as the reference path: pad rows (which include empty rows)
//      are **exactly 0**; ①' writes S outside the predicate as **exactly the −inf bit pattern** and
//      writes not a single word of `[live, col_cap)` or the pad rows; bit-identical when the
//      capacity C changes; bit-identical when split into row blocks; two dispatches on the same
//      input are bit-identical (determinism)
//   ④ fault injection: mutations that flatten the tree reduction / change the lanes' scan width
//      fail ① (proof that the check fires)
//
// The census check (decision 4 ③ of the same ADR — with the preset specified, ①' and ③' really
// run) is in gpu_state_execution_test.ts.

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import {
  referenceStateAttention,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";
import { stateColumnBase, stateLiveColumns, stateSliding } from "../src/kernels/state-attention.ts";
import { stateStatsPvEligible } from "../src/kernels/state-attention-stats-pv.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  assertMutated,
  caseColCap,
  halfScale,
  runStateAttention,
  seeded,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * The tolerance against the f64 reference (the same value and rationale as `STATE_TOLERANCE` in
 * gpu_state_attention_test.ts).
 */
const STATE_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/**
 * The A/B band between ③ and ③'.
 *
 * Derivation: both sum `live` products in f32 but in a different order (③ = ascending sequential,
 * ③' = 16-lane partial sums → a fixed-order tree). The difference is bounded by two f32 reduction
 * errors, with outputs of magnitude O(1). The measured worst over all `PARALLEL_CASES` below (the
 * reduction length `live` from 1 to 16,384) is **vs ③ maxAbs 2.38e-7 / vs the f64 reference
 * 3.99e-7** (2026-09-03 / RTX 3080 Ti, Vulkan). `atol = 5e-6` adds about 20× headroom to that (the
 * same value as the band against the f64 reference — if both sit in the same band around the f64
 * reference, their mutual difference is within twice that band). `rtol = 0` for the same reason as
 * STATE_TOLERANCE.
 * NOTE: the worst value comes from the cases on the **small** live side; adding the coverage cases
 * with live 4,096 / 16,384 does not move it (the weights spread over many columns, so the output
 * varies less). The main-session measurement (review 2026-09-03) shows the same trend up to
 * live 65,536 — ③' vs f64 1.8e-8 / ③ vs f64 3.6e-8 / ③' vs ③ 5.0e-8, all within 1/100 of the band
 * 5e-6.
 * NOTE: the measured values depend on the driver's use of fma, so on another adapter repeat the
 * whole "measure → add headroom" procedure (the test prints the worst value to stdout on every
 * run).
 */
const AB_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

/**
 * The A/B band between ① and ①' (checked on the output O — a difference in S reaches O through
 * softmax).
 *
 * The derivation has two steps:
 * 1. **The difference in S**: both sum the same `D` products (with the half scale applied on both
 *    the q side and the k side — generated in one place, `stateScoreFn` in
 *    src/kernels/state-attention.ts) in f32 but in a different order (① = sequential in
 *    ascending `d`,
 *    ①' = 16-lane partial sums → a fixed-order tree). In both, the f32 reduction error is bounded
 *    by `O(D · eps · Σ|term|)` (eps = 2⁻²³ ≈ 1.19e-7), and the difference is two of those. The grid
 *    in this file has `D ≤ 32` and terms of magnitude ≤ about 2.6
 *    (`|q| ≤ 1.87 · |k| ≤ 2.07 · scale²` with scale² = 1/√D), so `|S|` is O(10) or less and the
 *    difference in S is bounded by
 *    **about 1e-5**.
 * 2. **Propagation to O**: `p = exp(S − m)·inv`, so a difference δ in S moves the weight ratios
 *    only by a factor of `exp(δ) ≈ 1 + δ`, and the normalization (Σp = 1) cancels the common
 *    factor. So the difference in O stays on the order of `δ · max|V|` (`|V| ≤ 2.48`).
 *
 * The measured worst over all `PARALLEL_CASES` below is **vs ① maxAbs 3.58e-7 / vs the f64
 * reference 4.17e-7** (the difference in S itself is 4.77e-7; the preset's own combination
 * (①' + ③') vs the f64 reference is 3.58e-7 — 2026-09-06 / RTX 3080 Ti, Vulkan. The test prints it
 * on every run). `atol = 5e-6` equals ③''s A/B band, about 14× over the measurement and with
 * headroom over the bound above too (the same argument: if both sit in the f64 reference's band
 * 5e-6, their mutual difference is within twice that).
 * `rtol = 0` for the same reason as STATE_TOLERANCE.
 * NOTE: the measured values depend on the driver's use of fma, so on another adapter repeat the
 * whole "measure → add headroom" procedure.
 */
const QK_AB_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;
const SLOT_POISON_K = 9;
const SLOT_POISON_V = 400;

/**
 * The same poison rules as `makeInputs` in gpu_state_attention_test.ts (poison in non-resident rows
 * and pad rows).
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

/**
 * A grid that favours shapes where ③' takes a different path from ③: live crossing the lane count
 * 16 (partial sums spread over several lanes), live exactly a multiple of 16 / uneven, D not a
 * multiple of TILE_X (the path where lanes with `d ≥ D` join the barrier), several row blocks,
 * MQA 8:1 decode (Gemma 4 E2B type), sliding across a ring wrap.
 */
const PARALLEL_CASES: readonly StateCase[] = [
  {
    name: "full r1 decode P0 (live 1 = lane 0 only)",
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
  {
    name: "full r8 MQA decode live 18 (crosses 16)",
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
  {
    name: "full r8 MQA decode live 48 (exactly a multiple of 16) D 20 (uneven)",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 20,
    capacity: 64,
    window: 0,
    past: 47,
    query: 1,
  },
  {
    name: "full r8 MQA decode live 301",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 32,
    capacity: 512,
    window: 0,
    past: 300,
    query: 1,
  },
  // Coverage statement (extends the band's rationale to production live — review W-G1-1). Checks
  // that the band does not move by an order of magnitude with a four- / five-digit reduction length
  // `live`. The shape is the Gemma 4 E2B decode (MQA 8:1, D 32).
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
  {
    name: "full r8 MQA decode live 16384",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 32,
    capacity: 16384,
    window: 0,
    past: 16383,
    query: 1,
  },
  {
    name: "full r2 pad B2 D6 (prefill, with pad rows)",
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
  {
    name: "sliding W8 r1 ring wrap decode",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 1,
    depth: 4,
    capacity: 8,
    window: 8,
    past: 21,
    query: 1,
  },
  {
    name: "sliding W32 r2 prefill pad (live 36 crosses 16)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 6,
    depth: 8,
    capacity: 64,
    window: 32,
    past: 40,
    query: 5,
  },
  {
    name: "sliding W512 r8 MQA decode (Gemma 4 E2B type)",
    batch: 1,
    heads: 8,
    kvHeads: 1,
    chunkRows: 1,
    depth: 16,
    capacity: 512,
    window: 512,
    past: 700,
    query: 1,
  },
];

/**
 * Copies the slot rows into a container of capacity `capacity` (the spare rows are poison — the
 * values jump if they are read).
 */
const widenSlots = (spec: StateCase, inputs: StateInputs, capacity: number): StateInputs => {
  const kvPlanes = spec.batch * spec.kvHeads;
  const slotK = seeded(kvPlanes * capacity * spec.depth, () => SLOT_POISON_K);
  const slotV = seeded(kvPlanes * capacity * spec.depth, () => SLOT_POISON_V);
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < spec.capacity; row += 1) {
      for (let d = 0; d < spec.depth; d += 1) {
        const from = (plane * spec.capacity + row) * spec.depth + d;
        const to = (plane * capacity + row) * spec.depth + d;
        slotK[to] = inputs.slotK[from];
        slotV[to] = inputs.slotV[from];
      }
    }
  }
  return { ...inputs, slotK, slotV };
};

const padRows = (spec: StateCase): readonly number[] => {
  const rows: number[] = [];
  for (let row = spec.query; row < spec.chunkRows; row += 1) rows.push(row);
  return rows;
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const assertBitIdentical = (
  a: Float32Array<ArrayBuffer>,
  b: Float32Array<ArrayBuffer>,
  label: string,
): void => {
  assertEquals(a.length, b.length, `${label}: length`);
  const ab = bitsOf(a);
  const bb = bitsOf(b);
  for (let i = 0; i < ab.length; i += 1) {
    if (ab[i] !== bb[i]) {
      throw new Error(`${label}: element ${i} differs in bits (${a[i]} vs ${b[i]})`);
    }
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the states-form ③' KV-parallel reduction stays within the bands of both the f64 reference and ③ and keeps the structural guarantees (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      let worstRef = 0;
      let worstAb = 0;
      for (const spec of PARALLEL_CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const parallel = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          pvReduce: "parallel",
        });
        const sequential = await runStateAttention(gpu.device, spec, inputs, { cache });

        // ① the band of the f64 reference
        const ref = compareTensors({ dtype: "f32", data: parallel.out }, expected, STATE_TOLERANCE);
        assert(ref.pass, `${spec.name} vs reference: ${formatAllclose(ref)}`);
        worstRef = Math.max(worstRef, ref.maxAbsError);
        // ② the A/B band against ③
        const ab = compareTensors(
          { dtype: "f32", data: parallel.out },
          {
            dtype: "f32",
            data: sequential.out,
          },
          AB_TOLERANCE,
        );
        assert(ab.pass, `${spec.name} ③ vs ③': ${formatAllclose(ab)}`);
        worstAb = Math.max(worstAb, ab.maxAbsError);

        // ③ pad rows (which include empty rows) are exactly 0 (Object.is — lets neither −0 nor
        // NaN through)
        const rowsOf = spec.batch * spec.heads;
        for (let z = 0; z < rowsOf; z += 1) {
          for (const row of padRows(spec)) {
            for (let d = 0; d < spec.depth; d += 1) {
              const at = (z * spec.chunkRows + row) * spec.depth + d;
              assert(
                Object.is(parallel.out[at], 0),
                `${spec.name}: pad row (z ${z}, row ${row}, d ${d}) is not exactly 0 (${
                  parallel.out[at]
                })`,
              );
            }
          }
        }

        // ③ determinism (two dispatches on the same input are bit-identical)
        const again = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          pvReduce: "parallel",
        });
        assertBitIdentical(parallel.out, again.out, `${spec.name}: determinism`);

        // ③ capacity independence (a larger C leaves the output bit-identical — the flip side of
        // the work condition). The slots **copy the same rows into a wider container** (redrawing
        // seeded would move the plane offsets and give different values)
        if (!stateSliding(spec.window)) {
          const wide: StateCase = { ...spec, capacity: spec.capacity * 8 };
          const wideInputs = widenSlots(spec, inputs, wide.capacity);
          const wideOut = await runStateAttention(gpu.device, wide, wideInputs, {
            cache,
            pvReduce: "parallel",
          });
          assertBitIdentical(parallel.out, wideOut.out, `${spec.name}: capacity ×8`);
        }

        // ③ row-block independence (for M > 1, 1 block vs several are bit-identical)
        if (spec.chunkRows > 1) {
          const split: StateCase = {
            ...spec,
            rowsBlock: Math.max(1, Math.ceil(spec.chunkRows / 3)),
          };
          const one: StateCase = { ...spec, rowsBlock: spec.chunkRows };
          const a = await runStateAttention(gpu.device, one, inputs, {
            cache,
            pvReduce: "parallel",
          });
          const b = await runStateAttention(gpu.device, split, inputs, {
            cache,
            pvReduce: "parallel",
          });
          assertBitIdentical(a.out, b.out, `${spec.name}: row blocks 1 vs 3`);
        }
      }
      console.log(
        `③' measured worst: vs reference maxAbs ${worstRef.toExponential(2)} / vs ③ maxAbs ${worstAb.toExponential(
          2,
        )} (band ${STATE_TOLERANCE.atol} / ${AB_TOLERANCE.atol})`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

/** Replaces each `[from, to]` once; an anchor missing from the WGSL fails the test (no silent no-op). */
const replaceEach = (wgsl: string, ...pairs: readonly (readonly [string, string])[]): string =>
  pairs.reduce((text, [from, to]) => {
    assert(text.includes(from), `mutation anchor not found: ${from}`);
    return text.replace(from, to);
  }, wgsl);

/** Fault injection — each mutation fails ①'s band (proof that the check fires). */
test.skipIf(!GPU_AVAILABLE)(
  "states-form ③' fault injection (flatten the tree reduction / change the lane width) fails the reference band (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Unless live crosses the lane count, "lane 0's partial sum alone" would still match
      const spec = PARALLEL_CASES.find((c) => c.name.includes("live 301"))!;
      const inputs = makeInputs(spec);
      const expected = referenceStateAttention(refInput(spec, inputs));
      const mutations: readonly [string, (wgsl: string) => string][] = [
        [
          "flatten the tree reduction (skip the stride loop)",
          (wgsl) => wgsl.replace("var stride = 8u;", "var stride = 0u;"),
        ],
        [
          "double the lane scan's batch and tail strides (drops half of the columns)",
          // The scan is unrolled (attention_state_pv:v2 — perf-ledger K-91): a batch of 8 columns
          // steps 128 and the tail steps 16, so doubling the lane's stride doubles both steps. Each
          // anchor must exist, or the mutation would silently touch one loop only.
          (wgsl) =>
            replaceEach(
              wgsl,
              ["cl = cl + 128u;", "cl = cl + 256u;"],
              ["cl = cl + 16u;", "cl = cl + 32u;"],
            ),
        ],
      ];
      for (const [label, mutate] of mutations) {
        const result = await runStateAttention(gpu.device, spec, inputs, {
          pvReduce: "parallel",
          mutate: (kernel, wgsl) => {
            if (kernel !== "pv") return wgsl;
            const mutated = mutate(wgsl);
            assertMutated(wgsl, mutated, label);
            return mutated;
          },
        });
        const report = compareTensors(
          { dtype: "f32", data: result.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(
          report.pass,
          false,
          `${label}: the mutation passed the band (${formatAllclose(report)})`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Compares ①''s S against ① (**structure exact, only the values within a band**). Single-block
 * cases only (local row = global row; the S `runStateAttention` returns fits in one block).
 *
 * Four things are checked, all on the structural side that a tolerance cannot hide:
 *   ① inside the predicate both are finite and not poison (they are written) — the value
 *      difference is returned as the worst difference
 *   ② outside the predicate both are **exactly the −inf bit pattern** (the same bits at the same
 *      position)
 *   ③ the leftovers in `[live, col_cap)` stay poison in both (①''s column index is
 *      `wid.x·16 + lid.x`, so failing to stop the lanes of a partial tile writes here)
 *   ④ not a single word of the pad rows is written (direct evidence that the work scales with Q,
 *      not M)
 */
type ScoreCensus = { written: number; masked: number; beyond: number; skipped: number };

const assertQkScores = (
  spec: StateCase,
  parallel: Float32Array<ArrayBuffer>,
  sequential: Float32Array<ArrayBuffer>,
  census: ScoreCensus,
): number => {
  const colCap = caseColCap(spec);
  const base = stateColumnBase(spec.window, spec.past);
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  const rows = spec.chunkRows;
  const effRows = Math.min(spec.query, rows);
  const bitsP = bitsOf(parallel);
  const bitsS = bitsOf(sequential);
  let worst = 0;
  for (let plane = 0; plane < spec.batch * spec.heads; plane += 1) {
    for (let local = 0; local < effRows; local += 1) {
      const limit = spec.past + local;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        const inWindow = col <= limit && (!stateSliding(spec.window) || limit - col < spec.window);
        const at = (plane * rows + local) * colCap + cl;
        if (inWindow) {
          assert(
            Number.isFinite(parallel[at]) && parallel[at] !== STATE_S_POISON,
            `${spec.name}: ①' did not write (${local},${cl}) inside the predicate (${parallel[at]})`,
          );
          worst = Math.max(worst, Math.abs(parallel[at] - sequential[at]));
          census.written += 1;
        } else {
          assertEquals(
            [bitsP[at], bitsS[at]],
            [0xff800000, 0xff800000],
            `${spec.name}: (${local},${cl}) outside the predicate is not the −inf bit pattern (① vs ①')`,
          );
          census.masked += 1;
        }
      }
      for (let cl = live; cl < colCap; cl += 1) {
        const at = (plane * rows + local) * colCap + cl;
        assertEquals(
          parallel[at],
          STATE_S_POISON,
          `${spec.name}: ①' wrote (${local},${cl}) outside live (the lanes of a partial tile did not stop)`,
        );
        census.beyond += 1;
      }
    }
    for (let local = effRows; local < rows; local += 1) {
      for (let cl = 0; cl < colCap; cl += 1) {
        const at = (plane * rows + local) * colCap + cl;
        assertEquals(
          parallel[at],
          STATE_S_POISON,
          `${spec.name}: ①' wrote S of pad row (${local},${cl}) (it loops over M rows)`,
        );
        census.skipped += 1;
      }
    }
  }
  return worst;
};

/**
 * The band check of ①'. The grid reuses {@link PARALLEL_CASES} as is — it already covers `D` of
 * 4 / 6 / 8 (idle lanes), 16 (exactly one per lane), 20 (uneven = the path where lanes with
 * `d ≥ D` join the barrier) and 32 (two per lane), plus live a multiple of 16 / uneven, with pad
 * rows, several row blocks and sliding across a ring wrap (the requirements match ③''s grid).
 *
 * MUST: run with ③ **pinned to the reference path** (the preset switches both stages together,
 * but switching both here would make it impossible to tell whether a difference comes from ①' or
 * ③'). The preset's own combination (①' + ③') is checked last, against the f64 reference band
 * only.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the states-form ①' D-parallel reduction stays within the bands of both the f64 reference and ① and keeps the structural guarantees (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      let worstRef = 0;
      let worstAb = 0;
      let worstScore = 0;
      let worstSeat = 0;
      const census: ScoreCensus = { written: 0, masked: 0, beyond: 0, skipped: 0 };
      for (const spec of PARALLEL_CASES) {
        const inputs = makeInputs(spec);
        const expected = referenceStateAttention(refInput(spec, inputs));
        const parallel = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          qkReduce: "parallel",
        });
        const sequential = await runStateAttention(gpu.device, spec, inputs, { cache });

        // ① the band of the f64 reference
        const ref = compareTensors({ dtype: "f32", data: parallel.out }, expected, STATE_TOLERANCE);
        assert(ref.pass, `${spec.name} vs reference: ${formatAllclose(ref)}`);
        worstRef = Math.max(worstRef, ref.maxAbsError);
        // ② the A/B band against ①
        const ab = compareTensors(
          { dtype: "f32", data: parallel.out },
          {
            dtype: "f32",
            data: sequential.out,
          },
          QK_AB_TOLERANCE,
        );
        assert(ab.pass, `${spec.name} ① vs ①': ${formatAllclose(ab)}`);
        worstAb = Math.max(worstAb, ab.maxAbsError);

        // ③ the structure of S (bit equality of −inf outside the predicate; no writes outside
        // live or in pad rows). Cases split into row blocks return only the last block of S, so
        // only single blocks are checked
        if (spec.rowsBlock === undefined) {
          worstScore = Math.max(
            worstScore,
            assertQkScores(spec, parallel.scores, sequential.scores, census),
          );
        }

        // ④ determinism (two dispatches on the same input are bit-identical)
        const again = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          qkReduce: "parallel",
        });
        assertBitIdentical(parallel.out, again.out, `${spec.name}: determinism`);

        // ⑤ capacity independence (a larger C leaves the output bit-identical — the flip side of
        // the work condition)
        if (!stateSliding(spec.window)) {
          const wide: StateCase = { ...spec, capacity: spec.capacity * 8 };
          const wideInputs = widenSlots(spec, inputs, wide.capacity);
          const wideOut = await runStateAttention(gpu.device, wide, wideInputs, {
            cache,
            qkReduce: "parallel",
          });
          assertBitIdentical(parallel.out, wideOut.out, `${spec.name}: capacity ×8`);
        }

        // ⑥ row-block independence (for M > 1, 1 block vs several are bit-identical)
        if (spec.chunkRows > 1) {
          const split: StateCase = {
            ...spec,
            rowsBlock: Math.max(1, Math.ceil(spec.chunkRows / 3)),
          };
          const one: StateCase = { ...spec, rowsBlock: spec.chunkRows };
          const a = await runStateAttention(gpu.device, one, inputs, {
            cache,
            qkReduce: "parallel",
          });
          const b = await runStateAttention(gpu.device, split, inputs, {
            cache,
            qkReduce: "parallel",
          });
          assertBitIdentical(a.out, b.out, `${spec.name}: row blocks 1 vs 3`);
        }

        // ⑦ the preset's own combination (①' + ③') also sits in the f64 reference band
        const seat = await runStateAttention(gpu.device, spec, inputs, {
          cache,
          qkReduce: "parallel",
          pvReduce: "parallel",
        });
        const seatRef = compareTensors({ dtype: "f32", data: seat.out }, expected, STATE_TOLERANCE);
        assert(
          seatRef.pass,
          `${spec.name} preset parallel vs reference: ${formatAllclose(seatRef)}`,
        );
        worstSeat = Math.max(worstSeat, seatRef.maxAbsError);
      }
      // The checks fire (the grid has columns inside and outside the predicate, outside live and
      // pad rows)
      assert(
        census.written > 0 && census.masked > 0,
        `the S check never fires: ${JSON.stringify(census)}`,
      );
      assert(
        census.beyond > 0 && census.skipped > 0,
        `the S check never fires: ${JSON.stringify(census)}`,
      );
      console.log(
        `①' measured worst: vs reference maxAbs ${worstRef.toExponential(2)} / vs ① maxAbs ${worstAb.toExponential(
          2,
        )} / S difference ${worstScore.toExponential(2)} / preset ①'+③' vs reference ${worstSeat.toExponential(
          2,
        )} (band ${STATE_TOLERANCE.atol} / ${QK_AB_TOLERANCE.atol})`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

/** Fault injection — each mutation fails ①'s band (proof that the check fires). */
test.skipIf(!GPU_AVAILABLE)(
  "states-form ①' fault injection (flatten the tree reduction / change the lane width) fails the reference band (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // Unless D crosses the lane count, "lane 0's partial sum alone" would still match (D 32 =
      // two per lane)
      const spec = PARALLEL_CASES.find((c) => c.name.includes("live 301"))!;
      const inputs = makeInputs(spec);
      const expected = referenceStateAttention(refInput(spec, inputs));
      const mutations: readonly [string, (wgsl: string) => string][] = [
        [
          "flatten the tree reduction (skip the stride loop)",
          (wgsl) => wgsl.replace("var stride = 8u;", "var stride = 0u;"),
        ],
        [
          "set the lane scan width to 32 (drops half of D)",
          (wgsl) => wgsl.replace("d = d + 16u", "d = d + 32u"),
        ],
      ];
      for (const [label, mutate] of mutations) {
        const result = await runStateAttention(gpu.device, spec, inputs, {
          qkReduce: "parallel",
          mutate: (kernel, wgsl) => {
            if (kernel !== "qk") return wgsl;
            const mutated = mutate(wgsl);
            assertMutated(wgsl, mutated, label);
            return mutated;
          },
        });
        const report = compareTensors(
          { dtype: "f32", data: result.out },
          expected,
          STATE_TOLERANCE,
        );
        assertEquals(
          report.pass,
          false,
          `${label}: the mutation passed the band (${formatAllclose(report)})`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

// Stats/PV fusion (`statsPvFusion`) is a variant on top of the parallel seat (QK and PV both
// parallel), so its gates live beside the parallel-variant gates above.
// They compare in u32 against parallel before fusion and also keep the existing 5e-6 against the
// independent f64 reference.
//
// Inputs: the fusion gates keep their own sin-based inputs and poison values (20 / 400) instead
// of `makeInputs`, so these cases run on exactly the inputs they were written against.
const fusionInputs = (spec: StateCase): StateInputs => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = spec;
  const values = (count: number, salt: number): Float32Array<ArrayBuffer> =>
    seeded(count, (i) => Math.sin(i * 0.371 + salt) * 0.75);
  const inputs = {
    q: values(batch * heads * chunkRows * depth, 1),
    insK: values(batch * kvHeads * chunkRows * depth, 2),
    insV: values(batch * kvHeads * chunkRows * depth, 3),
    slotK: values(batch * kvHeads * capacity * depth, 4),
    slotV: values(batch * kvHeads * capacity * depth, 5),
  };
  const resident = new Set<number>();
  for (let col = window ? Math.max(0, past - window + 1) : 0; col < past; col++) {
    resident.add(window ? col % capacity : col);
  }
  for (let plane = 0; plane < batch * kvHeads; plane++) {
    for (let row = 0; row < capacity; row++) {
      if (resident.has(row)) continue;
      inputs.slotK.fill(20, (plane * capacity + row) * depth, (plane * capacity + row + 1) * depth);
      inputs.slotV.fill(
        400,
        (plane * capacity + row) * depth,
        (plane * capacity + row + 1) * depth,
      );
    }
    for (let row = query; row < chunkRows; row++) {
      inputs.insK.fill(
        20,
        (plane * chunkRows + row) * depth,
        (plane * chunkRows + row + 1) * depth,
      );
      inputs.insV.fill(
        400,
        (plane * chunkRows + row) * depth,
        (plane * chunkRows + row + 1) * depth,
      );
    }
  }
  return inputs;
};

describe("states attention row-stats/PV fusion", () => {
  it("respects the static eligibility bounds", () => {
    assert(stateStatsPvEligible(1, 1024));
    assert(stateStatsPvEligible(8, 1024));
    assertEquals(stateStatsPvEligible(9, 1024), false);
    assertEquals(stateStatsPvEligible(1, 1025), false);
  });

  it.skipIf(!GPU_AVAILABLE)(
    "returns the same values as parallel with ring, GQA, uneven D, pad and row splits",
    async () => {
      const gpu = await acquireGpu(),
        cache = new Map<string, GPUComputePipeline>();
      let maxAbs = 0;
      try {
        for (const depth of [1, 17, 256, 512]) {
          for (const [chunkRows, query, window, past] of [
            [1, 1, 0, 0],
            [4, 4, 0, 31],
            [8, 1, 0, 95],
            [8, 4, 1, 4095],
            [4, 1, 31, 4095],
            [8, 8, 127, 0xfffffff0],
            // live 701 (= past + query): the shape where the row stats' 256-lane grid-stride makes
            // three passes. The other six shapes have live ≤ 134 and finish in one pass, so nothing
            // would check on hardware that the multi-pass addition order matches ② (colCap 717 ≤
            // 1024 meets the eligibility condition).
            // The run time multiplies by the four depth loops, so this is the only long sequence.
            [1, 1, 0, 700],
          ]) {
            const spec: StateCase = {
              name: "fused",
              batch: 2,
              heads: 4,
              kvHeads: depth === 1 ? 4 : 1,
              chunkRows,
              query,
              window,
              past,
              depth,
              capacity: window ? window + 8 : past + chunkRows + 16,
            };
            const inputs = fusionInputs(spec);
            const ref = referenceStateAttention({ ...spec, ...inputs, scale: halfScale(depth) });
            const before = await runStateAttention(gpu.device, spec, inputs, {
              cache,
              qkReduce: "parallel",
              pvReduce: "parallel",
            });
            for (const rowsBlock of [chunkRows, 1]) {
              const after = await runStateAttention(gpu.device, { ...spec, rowsBlock }, inputs, {
                cache,
                qkReduce: "parallel",
                pvReduce: "parallel",
                statsPvFusion: true,
              });
              assertEquals(new Uint32Array(after.out.buffer), new Uint32Array(before.out.buffer));
              for (let i = 0; i < after.out.length; i++) {
                const error = Math.abs(after.out[i] - ref.data[i]);
                assert(error < 5e-6, "the existing states attention absolute tolerance");
                maxAbs = Math.max(maxAbs, error);
              }
            }
          }
        }
        console.log("stats/PV fusion vs f64 maxAbs", maxAbs);
      } finally {
        gpu.destroy();
      }
    },
  );

  it.skipIf(!GPU_AVAILABLE)(
    "preserves the NaN/Inf classification and the exact 0 of pad",
    async () => {
      const gpu = await acquireGpu(),
        cache = new Map<string, GPUComputePipeline>();
      try {
        const spec: StateCase = {
          name: "special",
          batch: 1,
          heads: 4,
          kvHeads: 1,
          chunkRows: 4,
          query: 1,
          window: 31,
          past: 40,
          capacity: 39,
          depth: 17,
        };
        for (const target of ["q", "insK", "insV", "slotK", "slotV"] as const) {
          const inputs = fusionInputs(spec);
          inputs[target].set([NaN, Infinity, -Infinity, 0, -0]);
          const before = await runStateAttention(gpu.device, spec, inputs, {
            cache,
            qkReduce: "parallel",
            pvReduce: "parallel",
          });
          const after = await runStateAttention(gpu.device, spec, inputs, {
            cache,
            qkReduce: "parallel",
            pvReduce: "parallel",
            statsPvFusion: true,
          });
          const a = new Uint32Array(after.out.buffer),
            b = new Uint32Array(before.out.buffer);
          for (let i = 0; i < a.length; i++) {
            if (Number.isNaN(before.out[i])) assert(Number.isNaN(after.out[i]));
            else assertEquals(a[i], b[i]);
            if (Math.floor(i / spec.depth) % spec.chunkRows >= spec.query) assertEquals(a[i], 0);
          }
        }
      } finally {
        gpu.destroy();
      }
    },
  );
});
