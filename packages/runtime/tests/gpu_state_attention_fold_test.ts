// Word-for-word identity gates of two bit-identical rewrites of the states-form decode attention
// kernels (ADR 0067 addendum of 2026-10-27):
//   - the folded ①' (F = r query heads of one kv plane per workgroup — perf-ledger K-55) against
//     ①' (F = 1, still in the product for every other plan)
//   - the unrolled ③' and fused stats + PV (`attention_state_pv:v2` / `attention_state_stats_pv:v2`
//     — perf-ledger K-91) against their v1 WGSL, kept test-only in fixtures/wgsl-baseline/ (the
//     generators no longer emit it)
//
// The contract is bit identity, not a band: every word of S / O is compared as u32, including the
// poison left in the words a kernel must not write, NaN and −inf. The identity rests on the backend
// compiling the same expression the same way, which WGSL does not guarantee — this file is the check
// to run on each backend. Order-only mutants (a reassociated product, two terms swapped) must
// mismatch on the same harness, or a zero-mismatch result proves nothing.
// NOTE: on Dawn / Vulkan / NVIDIA (2026-10-27) NaN payloads come out canonical and no output word is
// subnormal, consistent with flush-to-zero on this driver (observed by the independent check; not
// isolated), so the comparison cannot tell payloads or subnormal orderings apart there.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { readTextFile } from "@karume/test-support/fs";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  stateColumnBase,
  stateQkFoldHeads,
  statePvParallelWgsl,
  stateSliding,
} from "../src/kernels/state-attention.ts";
import { stateStatsPvWgsl } from "../src/kernels/state-attention-stats-pv.ts";
import {
  caseColCap,
  runStateAttention,
  runStatePv,
  runStateQk,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StateMutation,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** The shapes of the models that run these kernels (head dim, heads per kv plane), with B = 2. */
type Shape = Pick<StateCase, "heads" | "kvHeads" | "depth" | "window"> & { readonly name: string };

const E2B_FULL: Shape = { name: "e2b-full", heads: 8, kvHeads: 1, depth: 512, window: 0 };
const E4B_FULL: Shape = { name: "e4b-full", heads: 8, kvHeads: 2, depth: 512, window: 0 };
/** MiniCPM5 (16 q / 2 kv heads, r = 8 with two kv planes per batch row). */
const MINICPM5_FULL: Shape = {
  name: "minicpm5-full",
  heads: 16,
  kvHeads: 2,
  depth: 128,
  window: 0,
};
const E2B_SLIDING: Shape = { name: "e2b-sliding", heads: 8, kvHeads: 1, depth: 256, window: 512 };
const E4B_SLIDING: Shape = { name: "e4b-sliding", heads: 8, kvHeads: 2, depth: 256, window: 512 };
/** Non-GQA variants (kv plane = z) with a depth that leaves idle lanes in the last tile. */
const MHA_FULL: Shape = { name: "mha-full", heads: 2, kvHeads: 2, depth: 20, window: 0 };
const MHA_SLIDING: Shape = { name: "mha-sliding", heads: 2, kvHeads: 2, depth: 20, window: 32 };

/** Sliding slack of the distributed Gemma 4 slots (capacity = W + 8). */
const SLIDING_SLACK = 8;

/**
 * Lengths around the edges of the kernels: P = 0 (live = Q), live not a multiple of 8 / 16, live
 * just below / at / above one batch of the unrolled scan (8 columns × 16 lanes = 128), and long
 * contexts with many batches.
 */
const FULL_PASTS: readonly number[] = [0, 17, 127, 128, 1000, 4095];
/** Sliding: below the window, at its edge (live = W − 1 + Q) and past it (ring wrap). */
const SLIDING_PASTS: readonly number[] = [0, 17, 511, 512, 2047];

/**
 * (M, Q, rowsBlock): decode, verify buckets with and without pad rows, row blocks split in two
 * (the second block half pad rows / all pad rows).
 */
const ROWS: readonly (readonly [number, number, number])[] = [
  [1, 1, 1],
  [4, 4, 4],
  [4, 2, 4],
  [8, 8, 8],
  [8, 5, 8],
  [8, 6, 4],
  [8, 3, 4],
];

const caseOf = (shape: Shape, past: number, rows: readonly [number, number, number]): StateCase => {
  const [chunkRows, query, rowsBlock] = rows;
  return {
    name: `${shape.name} P=${past} M=${chunkRows} Q=${query} block=${rowsBlock}`,
    batch: 2,
    heads: shape.heads,
    kvHeads: shape.kvHeads,
    chunkRows,
    depth: shape.depth,
    capacity: stateSliding(shape.window) ? shape.window + SLIDING_SLACK : past + chunkRows + 5,
    window: shape.window,
    past,
    query,
    rowsBlock,
  };
};

/** xorshift32 in [0, 1) — deterministic, so a failure reproduces. */
const generator = (seed: number): (() => number) => {
  let x = (seed * 2654435761) >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x / 4294967296;
  };
};

const filled = (count: number, next: () => number): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(count);
  for (let i = 0; i < count; i += 1) data[i] = (next() * 2 - 1) * 3;
  return data;
};

/** A quiet NaN with a payload (the payload survives only where the backend propagates it). */
const NAN_PAYLOAD_BITS = 0x7fc12345;

/**
 * Inputs of one case. `nonFinite` adds, on query row 0:
 * - batch row 0, head 1: `q[d=0] = −inf` with every k at d = 0 positive → the whole S row is −inf
 *   (an empty row: stats (0, 0), output exactly 0)
 * - batch row 0, head 2: `q[d=0] = +inf` → S = +inf, `exp(inf − inf)` = NaN in the output row
 * - batch row 1, head 3: a NaN payload in one q element
 * - batch row 1: a NaN in the oldest live past K column and V row of kv plane 0 (every head of
 *   that plane)
 * - batch row 0: +inf / −inf in single elements of this step's V
 */
const inputsOf = (spec: StateCase, seed: number, nonFinite: boolean): StateInputs => {
  const next = generator(seed);
  const { batch, heads, kvHeads, chunkRows, depth, capacity } = spec;
  const q = filled(batch * heads * chunkRows * depth, next);
  const insK = filled(batch * kvHeads * chunkRows * depth, next);
  const insV = filled(batch * kvHeads * chunkRows * depth, next);
  const slotK = filled(batch * kvHeads * capacity * depth, next);
  const slotV = filled(batch * kvHeads * capacity * depth, next);
  if (nonFinite) {
    for (const k of [insK, slotK]) {
      for (let i = 0; i < k.length; i += depth) k[i] = Math.abs(k[i]) + 0.5;
    }
    const qAt = (b: number, h: number, d: number): number =>
      (b * heads + h) * chunkRows * depth + d;
    if (heads > 1) q[qAt(0, 1, 0)] = -Infinity;
    if (heads > 2) q[qAt(0, 2, 0)] = Infinity;
    // Written as bits: storing a NaN through a Float32Array may canonicalise its payload
    new Uint32Array(q.buffer)[qAt(1, Math.min(3, heads - 1), 11 % depth)] = NAN_PAYLOAD_BITS;
    const base = stateColumnBase(spec.window, spec.past);
    if (spec.past > base) {
      const row = stateSliding(spec.window) ? base % capacity : base;
      const plane = kvHeads; // batch row 1, kv head 0
      slotK[(plane * capacity + row) * depth + (7 % depth)] = Number.NaN;
      slotV[(plane * capacity + row) * depth + (5 % depth)] = Number.NaN;
    }
    insV[3 % depth] = Infinity;
    insV[depth + (4 % depth)] = -Infinity;
  }
  return { q, insK, insV, slotK, slotV };
};

/** Word classes seen in the reference outputs (the gate must meet each of them). */
type Census = { nan: number; negInf: number; zero: number; poison: number; finite: number };

const POISON_BITS = new Uint32Array(new Float32Array([STATE_S_POISON]).buffer)[0];

const tally = (census: Census, words: Uint32Array<ArrayBuffer>): void => {
  for (const w of words) {
    if (w === POISON_BITS) census.poison += 1;
    else if (w === 0xff800000) census.negInf += 1;
    else if ((w & 0x7f800000) === 0x7f800000 && (w & 0x007fffff) !== 0) census.nan += 1;
    else if ((w & 0x7fffffff) === 0) census.zero += 1;
    else census.finite += 1;
  }
};

/** The first mismatching word (index and both values), or undefined when the words agree. */
const firstMismatch = (
  a: Uint32Array<ArrayBuffer>,
  b: Uint32Array<ArrayBuffer>,
): string | undefined => {
  if (a.length !== b.length) return `length ${a.length} vs ${b.length}`;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return `word ${i}: 0x${a[i].toString(16)} vs 0x${b[i].toString(16)}`;
  }
  return undefined;
};

const countMismatches = (a: Uint32Array<ArrayBuffer>, b: Uint32Array<ArrayBuffer>): number =>
  a.reduce((n, w, i) => n + (w !== b[i] ? 1 : 0), 0);

const wordsOf = (data: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const variantSuffix = (spec: StateCase): string =>
  `${stateSliding(spec.window) ? "_sliding" : ""}${spec.heads !== spec.kvHeads ? "_gqa" : ""}`;

/** The v1 WGSL of ③' / the fused kernel for this case's variant (fixtures/wgsl-baseline/). */
const baselineOf = (kernel: "pv_par" | "stats_pv", spec: StateCase): Promise<string> =>
  readTextFile(
    new URL(
      `./fixtures/wgsl-baseline/attention_state_${kernel}${variantSuffix(spec)}.wgsl`,
      import.meta.url,
    ),
  );

/** Replaces `from` once; a missing anchor fails the test instead of leaving the kernel unmutated. */
const mustReplace = (wgsl: string, from: string, to: string): string => {
  assert(wgsl.includes(from), `mutation anchor not found: ${from}`);
  return wgsl.replace(from, to);
};

/** Replaces every `from`; a missing anchor fails the test as in {@link mustReplace}. */
const mustReplaceAll = (wgsl: string, from: string, to: string): string => {
  assert(wgsl.includes(from), `mutation anchor not found: ${from}`);
  return wgsl.replaceAll(from, to);
};

/** Swaps the terms of the first two columns of a batch (the same terms, added in another order). */
const swapFirstTwoTerms = (wgsl: string): string => {
  const first = "acc = acc + (exp(s[s_base + cl] - amax) * inv) * v0;";
  const second = "acc = acc + (exp(s[s_base + cl + 16u] - amax) * inv) * v1;";
  return mustReplace(
    mustReplace(mustReplace(wgsl, first, "@@FIRST@@"), second, first),
    "@@FIRST@@",
    second,
  );
};

describe.skipIf(!GPU_AVAILABLE)(
  "states attention kernels rewritten bit-identically (real GPU)",
  () => {
    it("the folded ①' writes the same S words as ①' (E2B / E4B / MiniCPM5 shapes, B = 2)", async () => {
      const gpu = await acquireGpu();
      const cache: StatePipelineCache = new Map();
      const census: Census = { nan: 0, negInf: 0, zero: 0, poison: 0, finite: 0 };
      let cases = 0;
      try {
        for (const shape of [E2B_FULL, E4B_FULL, MINICPM5_FULL]) {
          // The shapes this gate exists for are the ones the selection folds (at decode M = 1 on all
          // three; the identity is checked on every row shape below, folded by the selection or not)
          assertEquals(
            stateQkFoldHeads(false, shape.heads / shape.kvHeads, shape.depth, 1) > 1,
            true,
            shape.name,
          );
          for (const past of FULL_PASTS) {
            for (const rows of ROWS) {
              for (const nonFinite of [false, true]) {
                const spec = caseOf(shape, past, rows);
                const inputs = inputsOf(spec, 17 + cases, nonFinite);
                const label = `${spec.name}${nonFinite ? " non-finite" : ""}`;
                const reference = await runStateQk(gpu.device, spec, inputs, "parallel", { cache });
                const folded = await runStateQk(gpu.device, spec, inputs, "folded", { cache });
                assertEquals(folded.length, reference.length, `${label}: blocks`);
                for (const [block, words] of reference.entries()) {
                  const mismatch = firstMismatch(folded[block], words);
                  assertEquals(mismatch, undefined, `${label} block ${block}: ${mismatch}`);
                  tally(census, words);
                }
                cases += 1;
              }
            }
          }
        }
        // The grid met every word class the contract covers (written values, −inf outside the
        // predicate, NaN, and the poison of the words ①' must not write)
        assert(
          census.finite > 0 && census.negInf > 0 && census.nan > 0 && census.poison > 0,
          `S census: ${JSON.stringify(census)}`,
        );
        console.log(
          `folded ①' vs ①': ${cases} cases, 0 mismatching words, S census ${JSON.stringify(census)}`,
        );
      } finally {
        gpu.destroy();
      }
    });

    it("the unrolled ③' writes the same O words as the v1 ③' (full / sliding, GQA and not)", async () => {
      const gpu = await acquireGpu();
      const cache: StatePipelineCache = new Map();
      const census: Census = { nan: 0, negInf: 0, zero: 0, poison: 0, finite: 0 };
      let cases = 0;
      try {
        const shapes = [
          E2B_FULL,
          E4B_FULL,
          MINICPM5_FULL,
          E2B_SLIDING,
          E4B_SLIDING,
          MHA_FULL,
          MHA_SLIDING,
        ];
        for (const shape of shapes) {
          const pasts = stateSliding(shape.window) ? SLIDING_PASTS : FULL_PASTS;
          for (const past of pasts) {
            for (const rows of ROWS) {
              for (const nonFinite of [false, true]) {
                const spec = caseOf(shape, past, rows);
                const inputs = inputsOf(spec, 101 + cases, nonFinite);
                const label = `${spec.name}${nonFinite ? " non-finite" : ""}`;
                const baseline = await baselineOf("pv_par", spec);
                const toV1: StateMutation = (kernel, wgsl) => (kernel === "pv" ? baseline : wgsl);
                const v1 = await runStatePv(gpu.device, spec, inputs, "parallel", {
                  cache,
                  mutate: toV1,
                });
                const v2 = await runStatePv(gpu.device, spec, inputs, "parallel", { cache });
                const mismatch = firstMismatch(v2, v1);
                assertEquals(mismatch, undefined, `${label}: ${mismatch}`);
                tally(census, v1);
                cases += 1;
              }
            }
          }
        }
        // Exact zeros (pad rows, empty rows), NaN and finite outputs all occurred, and no poison is
        // left (③' writes every row of O)
        assert(
          census.finite > 0 && census.zero > 0 && census.nan > 0,
          `O census: ${JSON.stringify(census)}`,
        );
        assertEquals(census.poison, 0, "③' left an output word unwritten");
        console.log(
          `unrolled ③' vs v1: ${cases} cases, 0 mismatching words, O census ${JSON.stringify(census)}`,
        );
      } finally {
        gpu.destroy();
      }
    });

    it("the unrolled fused stats + PV writes the same O words as the v1 fused kernel", async () => {
      const gpu = await acquireGpu();
      const cache: StatePipelineCache = new Map();
      const census: Census = { nan: 0, negInf: 0, zero: 0, poison: 0, finite: 0 };
      let cases = 0;
      try {
        // Sliding layers take the fused kernel in the product; full shapes are run where their column
        // cap fits its eligibility (≤ 1024) as well, since the WGSL is the same for both
        const shapes = [
          E2B_SLIDING,
          E4B_SLIDING,
          MHA_SLIDING,
          E2B_FULL,
          E4B_FULL,
          MINICPM5_FULL,
          MHA_FULL,
        ];
        for (const shape of shapes) {
          const pasts = stateSliding(shape.window) ? SLIDING_PASTS : FULL_PASTS;
          for (const past of pasts) {
            for (const rows of ROWS) {
              for (const nonFinite of [false, true]) {
                const spec = caseOf(shape, past, rows);
                if (caseColCap(spec) > 1024) continue;
                const inputs = inputsOf(spec, 211 + cases, nonFinite);
                const label = `${spec.name}${nonFinite ? " non-finite" : ""}`;
                const baseline = await baselineOf("stats_pv", spec);
                const toV1: StateMutation = (kernel, wgsl) => (kernel === "pv" ? baseline : wgsl);
                const options = {
                  cache,
                  qkReduce: "parallel",
                  pvReduce: "parallel",
                  statsPvFusion: true,
                } as const;
                const v1 = await runStateAttention(gpu.device, spec, inputs, {
                  ...options,
                  mutate: toV1,
                });
                const v2 = await runStateAttention(gpu.device, spec, inputs, options);
                const mismatch = firstMismatch(wordsOf(v2.out), wordsOf(v1.out));
                assertEquals(mismatch, undefined, `${label}: ${mismatch}`);
                tally(census, wordsOf(v1.out));
                cases += 1;
              }
            }
          }
        }
        assert(
          census.finite > 0 && census.zero > 0 && census.nan > 0,
          `O census: ${JSON.stringify(census)}`,
        );
        assertEquals(census.poison, 0, "the fused kernel left an output word unwritten");
        console.log(
          `unrolled fused vs v1: ${cases} cases, 0 mismatching words, O census ${JSON.stringify(census)}`,
        );
      } finally {
        gpu.destroy();
      }
    });

    it("order-only mutants of the rewritten kernels mismatch (the word gate is not vacuous)", async () => {
      const gpu = await acquireGpu();
      try {
        const qkCases: readonly (readonly [string, Shape, (wgsl: string) => string])[] = [
          [
            "F8: head 3 multiplies q · (scale · kd) instead of (q · scale) · kd",
            E2B_FULL,
            (wgsl) =>
              wgsl.replaceAll(
                "acc3 = acc3 + (q[q_base3 + d] * params.scale) * kd;",
                "acc3 = acc3 + q[q_base3 + d] * (params.scale * kd);",
              ),
          ],
          [
            "F4: every head multiplies ((q · scale) · k) · scale instead of (q · scale) · (k · scale)",
            E4B_FULL,
            (wgsl) =>
              mustReplace(
                mustReplace(
                  mustReplaceAll(wgsl, ") * kd;", ") * kd * params.scale;"),
                  "let kd = slot_k[k_base + d] * params.scale;",
                  "let kd = slot_k[k_base + d];",
                ),
                "let kd = ins_k[k_base + d] * params.scale;",
                "let kd = ins_k[k_base + d];",
              ),
          ],
        ];
        for (const [label, shape, mutate] of qkCases) {
          const spec = caseOf(shape, 1000, [1, 1, 1]);
          const inputs = inputsOf(spec, 5, false);
          const mutation: StateMutation = (_kernel, wgsl) => {
            const mutated = mutate(wgsl);
            assertNotEquals(mutated, wgsl, `${label}: the mutation did not apply`);
            return mutated;
          };
          const reference = await runStateQk(gpu.device, spec, inputs, "parallel");
          const mutant = await runStateQk(gpu.device, spec, inputs, "folded", { mutate: mutation });
          const mismatches = countMismatches(mutant[0], reference[0]);
          assert(mismatches > 0, `${label}: the mutant matched ①' word for word`);
          console.log(`mutant '${label}': ${mismatches} of ${reference[0].length} S words differ`);
        }

        const pvCases: readonly (readonly [
          string,
          Shape,
          "pv_par" | "stats_pv",
          (wgsl: string) => string,
        ])[] = [
          ["③': the first two terms of a batch swapped", E2B_FULL, "pv_par", swapFirstTwoTerms],
          [
            "③': the batch's first term as exp · (inv · v)",
            E4B_FULL,
            "pv_par",
            (wgsl) =>
              mustReplace(
                wgsl,
                "acc = acc + (exp(s[s_base + cl] - amax) * inv) * v0;",
                "acc = acc + exp(s[s_base + cl] - amax) * (inv * v0);",
              ),
          ],
          [
            "fused: the first two terms of a batch swapped",
            E2B_SLIDING,
            "stats_pv",
            swapFirstTwoTerms,
          ],
        ];
        for (const [label, shape, kernel, mutate] of pvCases) {
          const spec = caseOf(shape, stateSliding(shape.window) ? 2047 : 4095, [1, 1, 1]);
          const inputs = inputsOf(spec, 9, false);
          const baseline = await baselineOf(kernel, spec);
          const product =
            kernel === "pv_par"
              ? statePvParallelWgsl(stateSliding(spec.window), true)
              : stateStatsPvWgsl(stateSliding(spec.window), true);
          // The baseline really is the other loop (a copy of the product would make the gates above
          // pass trivially)
          assertNotEquals(product, baseline, `${label}: the baseline equals the product WGSL`);
          assert(
            baseline.includes("for (var cl = lane; cl < live; cl = cl + 16u) {"),
            `${label}: v1 loop`,
          );
          const toV1: StateMutation = (k, wgsl) => (k === "pv" ? baseline : wgsl);
          const toMutant: StateMutation = (k, wgsl) => (k === "pv" ? mutate(wgsl) : wgsl);
          const run = (mutation: StateMutation): Promise<Uint32Array<ArrayBuffer>> =>
            kernel === "pv_par"
              ? runStatePv(gpu.device, spec, inputs, "parallel", { mutate: mutation })
              : runStateAttention(gpu.device, spec, inputs, {
                  qkReduce: "parallel",
                  pvReduce: "parallel",
                  statsPvFusion: true,
                  mutate: mutation,
                }).then((result) => wordsOf(result.out));
          const v1 = await run(toV1);
          const mutant = await run(toMutant);
          const mismatches = countMismatches(mutant, v1);
          assert(mismatches > 0, `${label}: the mutant matched v1 word for word`);
          console.log(`mutant '${label}': ${mismatches} of ${v1.length} O words differ`);
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);
