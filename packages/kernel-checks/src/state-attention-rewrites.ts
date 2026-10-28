/**
 * Check 3 — word-for-word identity of two bit-identical rewrites of the states-form decode attention
 * kernels (ADR 0067 addendum of 2026-10-27):
 *   - the folded ①' (F = r query heads of one kv plane per workgroup — perf-ledger K-55) against
 *     ①' (F = 1, still in the product for every other plan). F = 8 is selected in the product;
 *     F = 4 (E4B) is not — it differs on D3D12 — and stays here as the check perf-ledger K-95 needs
 *   - the unrolled ③' and fused stats + PV (`attention_state_pv:v2` / `attention_state_stats_pv:v2`
 *     — perf-ledger K-91) against their v1 WGSL, kept test-only in
 *     packages/runtime/tests/fixtures/wgsl-baseline/ (the generators no longer emit it)
 *
 * The contract is bit identity, not a band: every word of S / O is compared as u32, including the
 * poison left in the words a kernel must not write, NaN and −inf. The identity rests on the backend
 * compiling the same expression the same way, which WGSL does not guarantee — this is the check to
 * run on each backend. Order-only mutants (a reassociated product, two terms swapped) must mismatch
 * on the same harness, or a zero-mismatch result proves nothing.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import {
  STATE_QK_FOLD_CLASSES,
  stateColumnBase,
  stateLiveColumns,
  statePvParallelWgsl,
  stateQkFoldHeads,
  stateSliding,
} from "../../runtime/src/kernels/state-attention.ts";
import { stateStatsPvWgsl } from "../../runtime/src/kernels/state-attention-stats-pv.ts";
import pvParBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_pv_par.wgsl?raw";
import pvParGqaBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_pv_par_gqa.wgsl?raw";
import pvParSlidingBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_pv_par_sliding.wgsl?raw";
import pvParSlidingGqaBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_pv_par_sliding_gqa.wgsl?raw";
import statsPvBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_stats_pv.wgsl?raw";
import statsPvGqaBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_stats_pv_gqa.wgsl?raw";
import statsPvSlidingBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_stats_pv_sliding.wgsl?raw";
import statsPvSlidingGqaBaseline from "../../runtime/tests/fixtures/wgsl-baseline/attention_state_stats_pv_sliding_gqa.wgsl?raw";
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
} from "../../runtime/tests/helpers/state-dispatch.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareWords,
  type ExpectedCases,
  scopedGpuWork,
  throwIfAborted,
  wordsOf,
} from "./result.ts";

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

/**
 * The shapes the folded ①' is checked on, and what the product does with each: E2B and MiniCPM5 are
 * the shapes the selection folds (at decode M = 1); E4B is excluded (its F = 4 kernel differs on
 * D3D12 — perf-ledger K-95) and stays as the check K-95 needs.
 */
const FOLD_SHAPES: readonly { readonly shape: Shape; readonly use: "product" | "k95" }[] = [
  { shape: E2B_FULL, use: "product" },
  { shape: E4B_FULL, use: "k95" },
  { shape: MINICPM5_FULL, use: "product" },
];

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

const pastsOf = (shape: Shape): readonly number[] =>
  stateSliding(shape.window) ? SLIDING_PASTS : FULL_PASTS;

/** One case of a grid, with the seed of its inputs. */
type GridCase = {
  readonly shape: Shape;
  readonly spec: StateCase;
  readonly nonFinite: boolean;
  readonly seed: number;
};

/**
 * The cases of `shapes` × their lengths × {@link ROWS} (those `keep` admits) × finite / non-finite
 * inputs, in that order, seeded `seed`, `seed + 1`, … — one enumeration for an identity part and
 * for the mutants run on its cases, so both see the same inputs.
 */
const gridOf = (
  shapes: readonly Shape[],
  seed: number,
  keep: (spec: StateCase) => boolean = () => true,
): readonly GridCase[] =>
  shapes
    .flatMap((shape) =>
      pastsOf(shape).flatMap((past) =>
        ROWS.map((rows) => ({ shape, spec: caseOf(shape, past, rows) })),
      ),
    )
    .filter(({ spec }) => keep(spec))
    .flatMap((entry) => [false, true].map((nonFinite) => ({ ...entry, nonFinite })))
    .map((entry, index) => ({ ...entry, seed: seed + index }));

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
type WordCensus = {
  nan: number;
  negInf: number;
  zero: number;
  poison: number;
  finite: number;
};

const POISON_BITS = new Uint32Array(new Float32Array([STATE_S_POISON]).buffer)[0];

const emptyCensus = (): WordCensus => ({ nan: 0, negInf: 0, zero: 0, poison: 0, finite: 0 });

const tally = (census: WordCensus, words: Uint32Array<ArrayBuffer>): void => {
  for (const w of words) {
    if (w === POISON_BITS) census.poison += 1;
    else if (w === 0xff800000) census.negInf += 1;
    else if ((w & 0x7f800000) === 0x7f800000 && (w & 0x007fffff) !== 0) census.nan += 1;
    else if ((w & 0x7fffffff) === 0) census.zero += 1;
    else census.finite += 1;
  }
};

const countMismatches = (a: Uint32Array<ArrayBuffer>, b: Uint32Array<ArrayBuffer>): number =>
  a.reduce((n, w, i) => n + (w !== b[i] ? 1 : 0), 0);

type BaselineKernel = "pv_par" | "stats_pv";

/** The v1 WGSL of ③' / the fused kernel per variant (sliding, GQA). */
const BASELINES: Readonly<Record<BaselineKernel, Readonly<Record<string, string>>>> = {
  pv_par: {
    "": pvParBaseline,
    _gqa: pvParGqaBaseline,
    _sliding: pvParSlidingBaseline,
    _sliding_gqa: pvParSlidingGqaBaseline,
  },
  stats_pv: {
    "": statsPvBaseline,
    _gqa: statsPvGqaBaseline,
    _sliding: statsPvSlidingBaseline,
    _sliding_gqa: statsPvSlidingGqaBaseline,
  },
};

const variantSuffix = (spec: StateCase): string =>
  `${stateSliding(spec.window) ? "_sliding" : ""}${spec.heads !== spec.kvHeads ? "_gqa" : ""}`;

/** The v1 WGSL of ③' / the fused kernel for this case's variant (fixtures/wgsl-baseline/). */
const baselineOf = (kernel: BaselineKernel, spec: StateCase): string => {
  const suffix = variantSuffix(spec);
  const wgsl = BASELINES[kernel][suffix];
  if (wgsl === undefined) throw new Error(`no baseline WGSL for ${kernel}${suffix}`);
  return wgsl;
};

/** Replaces `from` once; a missing anchor throws instead of leaving the kernel unmutated. */
const mustReplace = (wgsl: string, from: string, to: string): string => {
  if (!wgsl.includes(from)) throw new Error(`mutation anchor not found: ${from}`);
  return wgsl.replace(from, to);
};

/** Replaces every `from`; a missing anchor throws as in {@link mustReplace}. */
const mustReplaceAll = (wgsl: string, from: string, to: string): string => {
  if (!wgsl.includes(from)) throw new Error(`mutation anchor not found: ${from}`);
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

const labelOf = (spec: StateCase, nonFinite: boolean): string =>
  `${spec.name}${nonFinite ? " non-finite" : ""}`;

const censusText = (census: WordCensus): string => JSON.stringify(census);

/** One class of the folded ①' grid: the product's choice for it and the F the check runs. */
export type FoldClass = {
  /** The `group` of the class's cases. */
  readonly group: string;
  readonly shape: string;
  /** `r` (query heads per kv plane). */
  readonly kvRepeat: number;
  readonly depth: number;
  /** `M`. */
  readonly chunkRows: number;
  /** F the check runs (`r`). */
  readonly checkedFoldHeads: number;
  /** F the product selects (`stateQkFoldHeads`; 1 = no fold). */
  readonly productFoldHeads: number;
  /** `product` = a shape the selection folds; `k95` = kept only as the check perf-ledger K-95 needs. */
  readonly use: "product" | "k95";
};

const groupOf = (shape: Shape, chunkRows: number): string => `${shape.name} M=${chunkRows}`;

/**
 * Every (shape, M) class of the folded ①' grid with the product's F for it. The product folds the
 * classes of `STATE_QK_FOLD_CLASSES`; the fold part requires every one of them to be in this grid
 * (a class the product folds but the grid misses would go unchecked).
 */
export const stateQkFoldClasses = (): readonly FoldClass[] =>
  FOLD_SHAPES.flatMap(({ shape, use }) =>
    [...new Set(ROWS.map(([chunkRows]) => chunkRows))].map((chunkRows) => {
      const kvRepeat = shape.heads / shape.kvHeads;
      return {
        group: groupOf(shape, chunkRows),
        shape: shape.name,
        kvRepeat,
        depth: shape.depth,
        chunkRows,
        checkedFoldHeads: kvRepeat,
        productFoldHeads: stateQkFoldHeads(false, kvRepeat, shape.depth, chunkRows),
        use,
      };
    }),
  );

/** The folded ①' grid (E2B / E4B / MiniCPM5 shapes, B = 2). */
const foldGrid = (): readonly GridCase[] =>
  gridOf(
    FOLD_SHAPES.map(({ shape }) => shape),
    17,
  );

/** The classes of `STATE_QK_FOLD_CLASSES` that no case of the grid runs. */
const uncoveredFoldClasses = (): readonly string[] => {
  const grid = stateQkFoldClasses();
  return STATE_QK_FOLD_CLASSES.filter(
    (selected) =>
      !grid.some(
        (checked) =>
          checked.kvRepeat === selected.kvRepeat &&
          checked.depth === selected.depth &&
          checked.chunkRows === selected.chunkRows,
      ),
  ).map(({ kvRepeat, depth, chunkRows }) => `r = ${kvRepeat}, D = ${depth}, M = ${chunkRows}`);
};

/** The cases each identity part of {@link checkStateAttentionRewrites} returns. */
export const STATE_ATTENTION_CASES = {
  "state-qk-fold": 252,
  "state-pv-unroll": 546,
  "state-stats-pv-unroll": 490,
} as const satisfies ExpectedCases;

/**
 * The order-only mutants {@link checkStateAttentionMutants} runs (one result each; the mutant
 * matrix has one row each).
 */
export const STATE_ATTENTION_MUTANTS = 5;

/** The folded ①' writes the same S words as ①' (E2B / E4B / MiniCPM5 shapes, B = 2). */
export const checkStateQkFold = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const cache: StatePipelineCache = new Map();
  const census = emptyCensus();
  const cases: CheckCase[] = [];
  const grid = foldGrid();
  for (const { shape, spec, nonFinite, seed } of grid) {
    throwIfAborted(options);
    const label = labelOf(spec, nonFinite);
    const inputs = inputsOf(spec, seed, nonFinite);
    const { reference, folded } = await scopedGpuWork(gpu, label, async () => ({
      reference: await runStateQk(gpu.device, spec, inputs, "parallel", { cache }),
      folded: await runStateQk(gpu.device, spec, inputs, "folded", { cache }),
    }));
    const blocks = Math.max(folded.length, reference.length);
    cases.push({
      label,
      group: groupOf(shape, spec.chunkRows),
      criteria: Array.from({ length: blocks }, (_, block) =>
        compareWords(
          `S words of block ${block}: folded vs ①' (u32)`,
          folded[block] ?? new Uint32Array(0),
          reference[block] ?? new Uint32Array(0),
        ),
      ),
    });
    for (const words of reference) tally(census, words);
    options.onProgress?.({ part: "state-qk-fold", done: cases.length, total: grid.length });
  }
  const uncovered = uncoveredFoldClasses();
  return {
    id: "state-qk-fold",
    title: "Folded ①' vs ①' (S words)",
    cases,
    requirements: [
      ...FOLD_SHAPES.map(({ shape, use }) => {
        const heads = stateQkFoldHeads(false, shape.heads / shape.kvHeads, shape.depth, 1);
        return {
          id: `selection-${shape.name}`,
          name: `the product ${use === "product" ? "folds" : "does not fold"} ${shape.name} at decode (M = 1)`,
          met: heads > 1 === (use === "product"),
          detail: `F = ${heads}`,
        };
      }),
      {
        id: "classes-covered",
        name: "every class the product folds (STATE_QK_FOLD_CLASSES) has cases in the grid",
        met: uncovered.length === 0,
        detail:
          uncovered.length === 0
            ? `${STATE_QK_FOLD_CLASSES.length} classes, all checked`
            : `in use, not checked: ${uncovered.join("; ")}`,
      },
      {
        // The grid met every word class the contract covers (written values, −inf outside the
        // predicate, NaN, and the poison of the words ①' must not write).
        id: "census",
        name: "the reference S met finite, −inf, NaN and poison words",
        met: census.finite > 0 && census.negInf > 0 && census.nan > 0 && census.poison > 0,
        detail: censusText(census),
      },
    ],
  };
};

const PV_SHAPES: readonly Shape[] = [
  E2B_FULL,
  E4B_FULL,
  MINICPM5_FULL,
  E2B_SLIDING,
  E4B_SLIDING,
  MHA_FULL,
  MHA_SLIDING,
];

/** The unrolled ③' grid (full / sliding, GQA and not). */
const pvGrid = (): readonly GridCase[] => gridOf(PV_SHAPES, 101);

/** O-side requirements: exact zeros, NaN and finite outputs occurred, and no word was left unwritten. */
const outputRequirements = (census: WordCensus, kernel: string): CheckPart["requirements"] => [
  {
    id: "census",
    name: "the reference O met finite, zero and NaN words",
    met: census.finite > 0 && census.zero > 0 && census.nan > 0,
    detail: censusText(census),
  },
  {
    id: "full-write",
    name: `${kernel} wrote every output word (no poison left)`,
    met: census.poison === 0,
    detail: `${census.poison} poison words`,
  },
];

/** The unrolled ③' writes the same O words as the v1 ③' (full / sliding, GQA and not). */
export const checkStatePvUnroll = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const cache: StatePipelineCache = new Map();
  const census = emptyCensus();
  const cases: CheckCase[] = [];
  const grid = pvGrid();
  for (const { spec, nonFinite, seed } of grid) {
    throwIfAborted(options);
    const label = labelOf(spec, nonFinite);
    const inputs = inputsOf(spec, seed, nonFinite);
    const baseline = baselineOf("pv_par", spec);
    const toV1: StateMutation = (kernel, wgsl) => (kernel === "pv" ? baseline : wgsl);
    const { v1, v2 } = await scopedGpuWork(gpu, label, async () => ({
      v1: await runStatePv(gpu.device, spec, inputs, "parallel", { cache, mutate: toV1 }),
      v2: await runStatePv(gpu.device, spec, inputs, "parallel", { cache }),
    }));
    cases.push({ label, criteria: [compareWords("O words: unrolled vs v1 (u32)", v2, v1)] });
    tally(census, v1);
    options.onProgress?.({ part: "state-pv-unroll", done: cases.length, total: grid.length });
  }
  return {
    id: "state-pv-unroll",
    title: "Unrolled ③' vs v1 ③' (O words)",
    cases,
    requirements: outputRequirements(census, "③'"),
  };
};

/**
 * Sliding layers take the fused kernel in the product; full shapes are run where their column cap
 * fits its eligibility (≤ 1024) as well, since the WGSL is the same for both.
 */
const STATS_PV_SHAPES: readonly Shape[] = [
  E2B_SLIDING,
  E4B_SLIDING,
  MHA_SLIDING,
  E2B_FULL,
  E4B_FULL,
  MINICPM5_FULL,
  MHA_FULL,
];

const STATS_PV_COLUMN_CAP = 1024;

/** The fused stats + PV grid: the cases whose column cap the fused kernel takes. */
const statsPvGrid = (): readonly GridCase[] =>
  gridOf(STATS_PV_SHAPES, 211, (spec) => caseColCap(spec) <= STATS_PV_COLUMN_CAP);

/** One fused stats + PV run (①' parallel, then the fused kernel) and its O words. */
const runStatsPv = async (
  gpu: GpuContext,
  spec: StateCase,
  inputs: StateInputs,
  run: { readonly cache?: StatePipelineCache; readonly mutate?: StateMutation },
): Promise<Uint32Array<ArrayBuffer>> =>
  wordsOf(
    (
      await runStateAttention(gpu.device, spec, inputs, {
        ...run,
        qkReduce: "parallel",
        pvReduce: "parallel",
        statsPvFusion: true,
      })
    ).out,
  );

/** The unrolled fused stats + PV writes the same O words as the v1 fused kernel. */
export const checkStateStatsPvUnroll = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const cache: StatePipelineCache = new Map();
  const census = emptyCensus();
  const cases: CheckCase[] = [];
  const grid = statsPvGrid();
  for (const { spec, nonFinite, seed } of grid) {
    throwIfAborted(options);
    const label = labelOf(spec, nonFinite);
    const inputs = inputsOf(spec, seed, nonFinite);
    const baseline = baselineOf("stats_pv", spec);
    const toV1: StateMutation = (kernel, wgsl) => (kernel === "pv" ? baseline : wgsl);
    const { v1, v2 } = await scopedGpuWork(gpu, label, async () => ({
      v1: await runStatsPv(gpu, spec, inputs, { cache, mutate: toV1 }),
      v2: await runStatsPv(gpu, spec, inputs, { cache }),
    }));
    cases.push({ label, criteria: [compareWords("O words: unrolled vs v1 (u32)", v2, v1)] });
    tally(census, v1);
    options.onProgress?.({ part: "state-stats-pv-unroll", done: cases.length, total: grid.length });
  }
  return {
    id: "state-stats-pv-unroll",
    title: "Unrolled fused stats + PV vs v1 (O words)",
    cases,
    requirements: outputRequirements(census, "the fused kernel"),
  };
};

/** One order-only mutant: how many of the case's words it changed (MUST be more than 0). */
export type MutantResult = {
  readonly label: string;
  readonly kernel: "qk-fold" | BaselineKernel;
  /** The one case it runs on. */
  readonly case: string;
  readonly differingWords: number;
  readonly words: number;
};

/**
 * Whether a mutant's code path runs on a case: `yes` (it MUST change a word), `no` (it MUST NOT —
 * a check of the classification itself) or `contraction` (it runs, but changes a word only when the
 * backend contracts `acc + x * y` into one fused multiply-add: counted, no requirement).
 */
type Acts = "yes" | "no" | "contraction";

type Mutant<Kernel> = {
  readonly label: string;
  readonly shape: Shape;
  readonly kernel: Kernel;
  readonly mutate: (wgsl: string) => string;
  /** The mutant matrix's classification of a case of {@link Mutant.shape}. */
  readonly acts: (spec: StateCase) => Acts;
  readonly actsWhen: string;
};

/** Live columns of a case (the scan length of ②③ and the column count ①' writes). */
const liveOf = (spec: StateCase): number => stateLiveColumns(spec.window, spec.past, spec.query);

/**
 * A swap of the first two terms of a batch: `acc` is exactly 0 in lane 0's first batch, where
 * without contraction `(0 + a) + b` equals `(0 + b) + a`, so the swap shows from the second batch on
 * (column 128 runs when 128 + 112 < live); with the multiply-add contracted the first batch differs
 * as well.
 */
const swapActs = (spec: StateCase): Acts =>
  liveOf(spec) >= 241 ? "yes" : liveOf(spec) >= 113 ? "contraction" : "no";
const SWAP_ACTS_WHEN =
  "yes: live >= 241 (a second 8-column batch on lane 0); contraction: 113 <= live < 241 (one batch, from acc = 0); no: live < 113";

const QK_MUTANTS: readonly Mutant<"qk-fold">[] = [
  {
    label: "F8: head 3 multiplies q · (scale · kd) instead of (q · scale) · kd",
    shape: E2B_FULL,
    kernel: "qk-fold",
    mutate: (wgsl) =>
      wgsl.replaceAll(
        "acc3 = acc3 + (q[q_base3 + d] * params.scale) * kd;",
        "acc3 = acc3 + q[q_base3 + d] * (params.scale * kd);",
      ),
    acts: () => "yes",
    actsWhen: "every case writes live >= 1 column of head 3",
  },
  {
    label:
      "F4: every head multiplies ((q · scale) · k) · scale instead of (q · scale) · (k · scale)",
    shape: E4B_FULL,
    kernel: "qk-fold",
    mutate: (wgsl) =>
      mustReplace(
        mustReplace(
          mustReplaceAll(wgsl, ") * kd;", ") * kd * params.scale;"),
          "let kd = slot_k[k_base + d] * params.scale;",
          "let kd = slot_k[k_base + d];",
        ),
        "let kd = ins_k[k_base + d] * params.scale;",
        "let kd = ins_k[k_base + d];",
      ),
    acts: () => "yes",
    actsWhen: "every case writes live >= 1 column of every head",
  },
];

const PV_MUTANTS: readonly Mutant<BaselineKernel>[] = [
  {
    label: "③': the first two terms of a batch swapped",
    shape: E2B_FULL,
    kernel: "pv_par",
    mutate: swapFirstTwoTerms,
    acts: swapActs,
    actsWhen: SWAP_ACTS_WHEN,
  },
  {
    label: "③': the batch's first term as exp · (inv · v)",
    shape: E4B_FULL,
    kernel: "pv_par",
    mutate: (wgsl) =>
      mustReplace(
        wgsl,
        "acc = acc + (exp(s[s_base + cl] - amax) * inv) * v0;",
        "acc = acc + exp(s[s_base + cl] - amax) * (inv * v0);",
      ),
    acts: (spec) => (liveOf(spec) >= 113 ? "yes" : "no"),
    actsWhen: "yes: live >= 113 (lane 0 runs at least one 8-column batch); no: live < 113",
  },
  {
    label: "fused: the first two terms of a batch swapped",
    shape: E2B_SLIDING,
    kernel: "stats_pv",
    mutate: swapFirstTwoTerms,
    acts: swapActs,
    actsWhen: SWAP_ACTS_WHEN,
  },
];

/** The ①' mutation of a QK mutant; a mutation that changes nothing throws. */
const qkMutation =
  ({ label, mutate }: Mutant<"qk-fold">): StateMutation =>
  (_kernel, wgsl) => {
    const mutated = mutate(wgsl);
    if (mutated === wgsl) throw new Error(`${label}: the mutation did not apply`);
    return mutated;
  };

/** The v1 WGSL of a PV mutant's kernel for `spec`, checked to really be the other loop. */
const checkedBaseline = ({ label, kernel }: Mutant<BaselineKernel>, spec: StateCase): string => {
  const baseline = baselineOf(kernel, spec);
  const product =
    kernel === "pv_par"
      ? statePvParallelWgsl(stateSliding(spec.window), spec.heads !== spec.kvHeads)
      : stateStatsPvWgsl(stateSliding(spec.window), spec.heads !== spec.kvHeads);
  // The baseline really is the other loop (a copy of the product would make the identity gates
  // pass trivially)
  if (product === baseline) throw new Error(`${label}: the baseline equals the product WGSL`);
  if (!baseline.includes("for (var cl = lane; cl < live; cl = cl + 16u) {")) {
    throw new Error(`${label}: the baseline is not the v1 loop`);
  }
  return baseline;
};

/** The O words of a PV mutant's kernel on `spec` with the kernel's WGSL replaced by `pv`. */
const runPvMutantKernel = (
  gpu: GpuContext,
  kernel: BaselineKernel,
  spec: StateCase,
  inputs: StateInputs,
  pv: (wgsl: string) => string,
  cache?: StatePipelineCache,
): Promise<Uint32Array<ArrayBuffer>> => {
  const mutate: StateMutation = (k, wgsl) => (k === "pv" ? pv(wgsl) : wgsl);
  const run = cache === undefined ? { mutate } : { cache, mutate };
  return kernel === "pv_par"
    ? runStatePv(gpu.device, spec, inputs, "parallel", run)
    : runStatsPv(gpu, spec, inputs, run);
};

/** Order-only mutants of the rewritten kernels (each MUST mismatch — the word gate is not vacuous). */
export const checkStateAttentionMutants = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<readonly MutantResult[]> => {
  const results: MutantResult[] = [];
  const total = QK_MUTANTS.length + PV_MUTANTS.length;
  for (const qkMutant of QK_MUTANTS) {
    const { label, shape } = qkMutant;
    throwIfAborted(options);
    const spec = caseOf(shape, 1000, [1, 1, 1]);
    const inputs = inputsOf(spec, 5, false);
    const mutation = qkMutation(qkMutant);
    const { reference, mutant } = await scopedGpuWork(gpu, label, async () => ({
      reference: await runStateQk(gpu.device, spec, inputs, "parallel"),
      mutant: await runStateQk(gpu.device, spec, inputs, "folded", { mutate: mutation }),
    }));
    results.push({
      label,
      kernel: "qk-fold",
      case: spec.name,
      differingWords: countMismatches(mutant[0], reference[0]),
      words: reference[0].length,
    });
    options.onProgress?.({ part: "state-mutants", done: results.length, total });
  }
  for (const pvMutant of PV_MUTANTS) {
    const { label, shape, kernel, mutate } = pvMutant;
    throwIfAborted(options);
    const spec = caseOf(shape, stateSliding(shape.window) ? 2047 : 4095, [1, 1, 1]);
    const inputs = inputsOf(spec, 9, false);
    const baseline = checkedBaseline(pvMutant, spec);
    const { v1, mutant } = await scopedGpuWork(gpu, label, async () => ({
      v1: await runPvMutantKernel(gpu, kernel, spec, inputs, () => baseline),
      mutant: await runPvMutantKernel(gpu, kernel, spec, inputs, mutate),
    }));
    results.push({
      label,
      kernel,
      case: spec.name,
      differingWords: countMismatches(mutant, v1),
      words: v1.length,
    });
    options.onProgress?.({ part: "state-mutants", done: results.length, total });
  }
  return results;
};

/**
 * One mutant run on every case of its shape in its kernel's identity grid. Where the mutated code
 * path runs, every case MUST mismatch; where it does not, none may (or the classification is
 * wrong). A mutant that changes words on some cases only would leave the identity comparison
 * vacuous on the others — the one-case mutant check cannot see that.
 */
export type MutantMatrixRow = {
  readonly label: string;
  readonly kernel: MutantResult["kernel"];
  readonly shape: string;
  readonly actsWhen: string;
  readonly cases: number;
  /** Cases the mutated path runs on. */
  readonly acting: number;
  /** Acting cases where the mutant matched word for word (MUST be none). */
  readonly actingMatched: readonly string[];
  /** Cases the mutated path does not run on. */
  readonly idle: number;
  /** Idle cases where the mutant changed a word (MUST be none). */
  readonly idleMismatched: readonly string[];
  /** Cases where only a contracted multiply-add makes the mutant change a word. */
  readonly contraction: number;
  /** How many of those mismatched (whether this backend contracts; no requirement). */
  readonly contractionMismatched: number;
};

type MatrixPlan = {
  readonly mutant: Mutant<"qk-fold"> | Mutant<BaselineKernel>;
  readonly grid: readonly GridCase[];
};

/** Each mutant with the cases of its shape in its kernel's identity grid (same seeds). */
const matrixPlan = (): readonly MatrixPlan[] => {
  const ofShape = (grid: readonly GridCase[], shape: Shape): readonly GridCase[] =>
    grid.filter((entry) => entry.shape === shape);
  return [
    ...QK_MUTANTS.map((mutant) => ({ mutant, grid: ofShape(foldGrid(), mutant.shape) })),
    ...PV_MUTANTS.map((mutant) => ({
      mutant,
      grid: ofShape(mutant.kernel === "pv_par" ? pvGrid() : statsPvGrid(), mutant.shape),
    })),
  ];
};

/** The words of a mutant that differ from the unmutated kernel on one case. */
const mutantDifference = async (
  gpu: GpuContext,
  mutant: Mutant<"qk-fold"> | Mutant<BaselineKernel>,
  spec: StateCase,
  inputs: StateInputs,
  cache: StatePipelineCache,
): Promise<number> => {
  if (mutant.kernel === "qk-fold") {
    const reference = await runStateQk(gpu.device, spec, inputs, "parallel", { cache });
    const mutated = await runStateQk(gpu.device, spec, inputs, "folded", {
      cache,
      mutate: qkMutation(mutant),
    });
    return reference.reduce(
      (total, words, block) =>
        total + compareWords("", mutated[block] ?? new Uint32Array(0), words).failures,
      0,
    );
  }
  const baseline = checkedBaseline(mutant, spec);
  const v1 = await runPvMutantKernel(gpu, mutant.kernel, spec, inputs, () => baseline, cache);
  const mutated = await runPvMutantKernel(gpu, mutant.kernel, spec, inputs, mutant.mutate, cache);
  return compareWords("", mutated, v1).failures;
};

/** Runs every mutant on every case of its shape (the browser runbook's matrix; slower than the one-case check). */
export const checkStateAttentionMutantMatrix = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<readonly MutantMatrixRow[]> => {
  const plan = matrixPlan();
  const total = plan.reduce((sum, { grid }) => sum + grid.length, 0);
  const rows: MutantMatrixRow[] = [];
  let done = 0;
  for (const { mutant, grid } of plan) {
    // Keyed by the final WGSL, so the mutated pipeline is never served for the unmutated run
    const cache: StatePipelineCache = new Map();
    const actingMatched: string[] = [];
    const idleMismatched: string[] = [];
    let acting = 0;
    let idle = 0;
    let contraction = 0;
    let contractionMismatched = 0;
    for (const { spec, nonFinite, seed } of grid) {
      throwIfAborted(options);
      const label = labelOf(spec, nonFinite);
      const inputs = inputsOf(spec, seed, nonFinite);
      const differing = await scopedGpuWork(gpu, `${mutant.label}: ${label}`, () =>
        mutantDifference(gpu, mutant, spec, inputs, cache),
      );
      const acts = mutant.acts(spec);
      if (acts === "yes") {
        acting += 1;
        if (differing === 0) actingMatched.push(label);
      } else if (acts === "no") {
        idle += 1;
        if (differing > 0) idleMismatched.push(label);
      } else {
        contraction += 1;
        if (differing > 0) contractionMismatched += 1;
      }
      done += 1;
      options.onProgress?.({ part: "state-mutant-matrix", done, total });
    }
    rows.push({
      label: mutant.label,
      kernel: mutant.kernel,
      shape: mutant.shape.name,
      actsWhen: mutant.actsWhen,
      cases: grid.length,
      acting,
      actingMatched,
      idle,
      idleMismatched,
      contraction,
      contractionMismatched,
    });
  }
  return rows;
};

/** Check 3 as one run: the three identity parts, the mutants (one case and matrix) and the fold classes. */
export type StateAttentionRewritesReport = {
  readonly parts: readonly CheckPart[];
  readonly mutants: readonly MutantResult[];
  readonly mutantMatrix: readonly MutantMatrixRow[];
  readonly foldClasses: readonly FoldClass[];
};

export const checkStateAttentionRewrites = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<StateAttentionRewritesReport> => {
  const qkFold = await checkStateQkFold(gpu, options);
  const pvUnroll = await checkStatePvUnroll(gpu, options);
  const statsPvUnroll = await checkStateStatsPvUnroll(gpu, options);
  const mutants = await checkStateAttentionMutants(gpu, options);
  const mutantMatrix = await checkStateAttentionMutantMatrix(gpu, options);
  return {
    parts: [qkFold, pvUnroll, statsPvUnroll],
    mutants,
    mutantMatrix,
    foldClasses: stateQkFoldClasses(),
  };
};
