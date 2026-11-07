/**
 * Check 8 — the bidirectional block of ①ₜ (`GenerationRun.bidirectionalRows`, ADR 0144 decision 3).
 *
 * The block changes the predicate and nothing else, so the oracle is the same kernel with a predicate
 * that is always true (every live score, in the same words) and a host mirror of the predicate that
 * replaces the columns outside it with −inf. A word that differs is a predicate that differs.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import {
  caseColCap,
  runStateQk,
  type StateCase,
  type StateMutation,
  type StatePipelineCache,
} from "../../runtime/tests/helpers/state-dispatch.ts";
import { inputsOf, type Shape } from "./state-attention-rewrites.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareWords,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

/** 12B's sliding layers, and a small window that a block fills exactly. */
const TWELVE_B_SLIDING: Shape = {
  name: "12b-sliding",
  heads: 16,
  kvHeads: 8,
  depth: 256,
  window: 1024,
};
const SMALL_SLIDING: Shape = { name: "small-sliding", heads: 2, kvHeads: 2, depth: 20, window: 32 };

const PASTS: readonly number[] = [0, 17, 1000, 5000];

type Rows = {
  readonly chunkRows: number;
  readonly query: number;
  readonly rowsBlock: number;
  readonly start: number;
  readonly end: number;
};

/** A block inside a full chunk, one with pad rows after it, a split row block, one at row 0. */
const ROWS: readonly Rows[] = [
  { chunkRows: 16, query: 16, rowsBlock: 16, start: 2, end: 14 },
  { chunkRows: 64, query: 48, rowsBlock: 64, start: 5, end: 37 },
  { chunkRows: 64, query: 48, rowsBlock: 16, start: 5, end: 37 },
  { chunkRows: 32, query: 32, rowsBlock: 32, start: 0, end: 32 },
];

/** The soft tokens of one image in a chunk of the product's length, over a wrapped ring. */
const IMAGE_CHUNK: Rows = { chunkRows: 768, query: 400, rowsBlock: 256, start: 100, end: 380 };

const specOf = (shape: Shape, past: number, rows: Rows): StateCase => ({
  name: `${shape.name} P=${past} M=${rows.chunkRows} Q=${rows.query} rows ${rows.rowsBlock} block [${rows.start}, ${rows.end})`,
  batch: 1,
  heads: shape.heads,
  kvHeads: shape.kvHeads,
  chunkRows: rows.chunkRows,
  depth: shape.depth,
  capacity: shape.window + 8,
  window: shape.window,
  past,
  query: rows.query,
  rowsBlock: rows.rowsBlock,
});

export const STATE_BIDIRECTIONAL_CASES = {
  "state-bidirectional": 2 * PASTS.length * ROWS.length + 1,
};

const NEG_INF_WORD = 0xff800000;

const TEXT_PREDICATE = /fn in_window\(col: u32, limit: u32\) -> bool \{[^}]*\}/;

/** The kernel with every live score kept: the predicate decides nothing. */
const everyScore: StateMutation = (_kernel, wgsl) => {
  const mutated = wgsl.replace(
    TEXT_PREDICATE,
    "fn in_window(col: u32, limit: u32) -> bool {\n  return true;\n}",
  );
  if (mutated === wgsl) throw new Error("the kernel has no in_window to replace");
  return mutated;
};

/** The words of one row block with the host's predicate applied to {@link everyScore}'s words. */
const expectedWords = (
  spec: StateCase,
  rows: Rows,
  rowOffset: number,
  all: Uint32Array<ArrayBuffer>,
): { readonly words: Uint32Array<ArrayBuffer>; readonly ahead: number } => {
  const { past, query, window } = spec;
  const planes = spec.batch * spec.heads;
  const colCap = caseColCap(spec);
  const held = Math.min(past, window - 1);
  const live = held + query;
  const base = past - held;
  const blockRows = Math.min(rows.rowsBlock, spec.chunkRows - rowOffset);
  const validRows = Math.max(0, Math.min(blockRows, query - rowOffset));
  const lo = past + rows.start;
  const hi = past + rows.end;
  const words = all.slice();
  let ahead = 0;
  for (let plane = 0; plane < planes; plane += 1) {
    for (let local = 0; local < validRows; local += 1) {
      const limit = past + rowOffset + local;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        const causal = col <= limit && limit - col < window;
        const block = limit >= lo && limit < hi && col >= lo && col < hi;
        if (block && col > limit) ahead += 1;
        if (!causal && !block) {
          words[(plane * blockRows + local) * colCap + cl] = NEG_INF_WORD;
        }
      }
    }
  }
  return { words, ahead };
};

const runCase = async (
  gpu: GpuContext,
  shape: Shape,
  past: number,
  rows: Rows,
  seed: number,
  cache: StatePipelineCache,
  mutate?: StateMutation,
): Promise<{ readonly case: CheckCase; readonly ahead: number }> => {
  const spec = specOf(shape, past, rows);
  const inputs = inputsOf(spec, seed, false);
  return await scopedGpuWork(gpu, spec.name, async () => {
    const all = await runStateQk(gpu.device, spec, inputs, "tiled", {
      cache,
      mutate: everyScore,
    });
    const actual = await runStateQk(gpu.device, spec, inputs, "tiled", {
      cache,
      bidirectionalRows: { start: rows.start, end: rows.end },
      ...(mutate === undefined ? {} : { mutate }),
    });
    let ahead = 0;
    const criteria = all.map((words, block) => {
      const expected = expectedWords(spec, rows, block * rows.rowsBlock, words);
      ahead += expected.ahead;
      return compareWords(
        `S words of row block ${block}`,
        actual[block] ?? new Uint32Array(0),
        expected.words,
      );
    });
    return { case: { label: spec.name, criteria }, ahead };
  });
};

export const checkStateBidirectional = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const cache: StatePipelineCache = new Map();
  const cases: CheckCase[] = [];
  let ahead = 0;
  let seed = 910_000;
  const grid = [
    ...[TWELVE_B_SLIDING, SMALL_SLIDING].flatMap((shape) =>
      PASTS.flatMap((past) => ROWS.map((rows) => ({ shape, past, rows }))),
    ),
    { shape: TWELVE_B_SLIDING, past: 1000, rows: IMAGE_CHUNK },
  ];
  for (const { shape, past, rows } of grid) {
    throwIfAborted(options);
    seed += 1;
    const result = await runCase(gpu, shape, past, rows, seed, cache);
    cases.push(result.case);
    ahead += result.ahead;
    options.onProgress?.({ part: "state-bidirectional", done: cases.length, total: grid.length });
  }
  return {
    id: "state-bidirectional",
    title: "Bidirectional block of the sliding ①ₜ",
    cases,
    requirements: [
      {
        id: "ahead",
        name: "the block lets rows see columns after them",
        met: ahead > 0,
        detail: `${ahead} scores`,
      },
    ],
  };
};

/** A mutant of the check: the words of its output that differ from the correct form's. */
export type StateBidirectionalMutant = { readonly label: string; readonly failures: number };

const MUTANTS: readonly { readonly label: string; readonly from: string; readonly to: string }[] = [
  {
    label: "the block's end is one row late",
    from: "limit < lengths.block_hi",
    to: "limit <= lengths.block_hi",
  },
  {
    label: "the block's first row sees nothing ahead",
    from: "limit >= lengths.block_lo",
    to: "limit > lengths.block_lo",
  },
  {
    label: "the block's rows see one column past it",
    from: "col < lengths.block_hi",
    to: "col <= lengths.block_hi",
  },
  {
    label: "columns of the block see rows outside it",
    from: "limit >= lengths.block_lo && limit < lengths.block_hi &&",
    to: "",
  },
];

/** Each predicate fault against the correct form's words: all must differ. */
export const stateBidirectionalMutants = async (
  gpu: GpuContext,
): Promise<readonly StateBidirectionalMutant[]> => {
  const cache: StatePipelineCache = new Map();
  const results: StateBidirectionalMutant[] = [];
  for (const { label, from, to } of MUTANTS) {
    const mutate: StateMutation = (_kernel, wgsl) => {
      const mutated = wgsl.replace(from, to);
      if (mutated === wgsl) throw new Error(`${label}: nothing to mutate`);
      return mutated;
    };
    const { case: run } = await runCase(gpu, SMALL_SLIDING, 17, ROWS[1], 920_000, cache, mutate);
    results.push({
      label,
      failures: run.criteria.reduce((total, criterion) => total + criterion.failures, 0),
    });
  }
  return results;
};
