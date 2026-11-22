/**
 * The band of plain attention (`attrs.band` — ADR 0155: key column `n` hidden from query row `m`
 * when `|m − n| > band`, checked inside ①QK from positions) against the plan it stands for: the
 * same node fed the band as an additive `[1,1,M,N]` mask of 0 / −inf, on the same backend. Every
 * output word identical (class E, the mirror being the mask plan).
 *
 * The two plans differ in one epilogue line of ①QK: the band **replaces** a hidden score by −inf
 * (`select`), the mask **adds** 0 or −inf to it. On a finite score they agree except in the sign
 * of a zero (`−0 + 0 = +0`), which ② and ③ only use as `exp(S − m)`; so the output is the same
 * words when the backend evaluates both the same way, and this check is the evidence per backend.
 *
 * The grid covers both ①QK forms (scalar, vec4), GQA and not, T not a multiple of the 128-row tile,
 * a band below T, at or above T (nothing hidden), far above it (the clamped band word) and 0 (the
 * diagonal only), several row blocks through the budget `attentionRowBlockMaxBytes` (the absolute
 * row of the predicate), S stored in f16 (`attentionScoreStorage: "f16"`), and query rows M unequal
 * to key columns N both ways — M > N with rows that see no column, whose output must be 0. Every q
 * holds rows of −0 and every k columns of −0: under IEEE arithmetic their scores are +0 in both
 * plans (the accumulator starts at +0, and `+0 + (±0)` is +0); on a backend that does not keep the
 * sign of a zero they may be −0, which is where the select and the add differ, in the sign only.
 * Each case also counts the ①QK dispatches of each side by its key (`:band` and `:mask`, with
 * `:rwa` where the case splits, `:s16` where S is stored in f16 and `:c16` under f16 compute), so a
 * side that did not run the kernel it stands for — a silent fallback to f32 storage included —
 * fails.
 *
 * NOTE: f16 compute (`attentionCompute: "f16"`) needs a device acquired with shader-f16, which the
 * workbench does not request; its cases are in the runtime's node test.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import type { Session, SessionOptions } from "../../runtime/src/runtime/executor.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  openGraphModel,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { openTwinSession } from "./fusion-twins/session.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareCount,
  compareWords,
  type ExpectedCases,
  scopedGpuWork,
  throwIfAborted,
  wordsOf,
} from "./result.ts";

/** One case of the grid: an attention shape (self-attention unless `keys` is set), its band and the
 * Session options. */
export type AttentionBandCase = {
  readonly name: string;
  readonly batch: number;
  readonly heads: number;
  readonly kvHeads: number;
  /** Query rows M. */
  readonly tokens: number;
  /** Key columns N (absent = `tokens`: self-attention). */
  readonly keys?: number;
  readonly depth: number;
  readonly band: number;
  /** Row blocks the budget splits the node into (1 = no budget). */
  readonly blocks: number;
  /** S stored in f16 (`attentionScoreStorage: "f16"`). */
  readonly s16?: true;
  /** f16 compute (`attentionCompute: "f16"` — node test only, needs shader-f16). */
  readonly c16?: true;
};

/** The key columns N of a case. */
export const attentionBandKeys = (testCase: AttentionBandCase): number =>
  testCase.keys ?? testCase.tokens;

/** The f32 / s16 grid (N = 37 or 21: scalar ①QK, N % 4 != 0; N = 40 and 136: vec4, D = 16). */
export const ATTENTION_BAND_GRID: readonly AttentionBandCase[] = [
  {
    name: "scalar H2 T37 D12 band 5",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 37,
    depth: 12,
    band: 5,
    blocks: 1,
  },
  {
    name: "scalar GQA B2 H4:2 T37 D12 band 3",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    tokens: 37,
    depth: 12,
    band: 3,
    blocks: 1,
  },
  {
    name: "vec4 H2 T40 D16 band 7",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 40,
    depth: 16,
    band: 7,
    blocks: 1,
  },
  {
    name: "vec4 GQA H6:2 T136 D16 band 9",
    batch: 1,
    heads: 6,
    kvHeads: 2,
    tokens: 136,
    depth: 16,
    band: 9,
    blocks: 1,
  },
  {
    name: "vec4 H2 T40 D16 band 40 (nothing hidden)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 40,
    depth: 16,
    band: 40,
    blocks: 1,
  },
  {
    name: "scalar H2 T37 D12 band 2^40 (clamped word)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 37,
    depth: 12,
    band: 2 ** 40,
    blocks: 1,
  },
  {
    name: "vec4 MQA H2:1 T40 D16 band 0 (diagonal)",
    batch: 1,
    heads: 2,
    kvHeads: 1,
    tokens: 40,
    depth: 16,
    band: 0,
    blocks: 1,
  },
  {
    name: "scalar GQA B2 H4:2 T37 D12 band 5 / 3 row blocks",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    tokens: 37,
    depth: 12,
    band: 5,
    blocks: 3,
  },
  {
    name: "vec4 GQA H6:2 T136 D16 band 9 / 3 row blocks",
    batch: 1,
    heads: 6,
    kvHeads: 2,
    tokens: 136,
    depth: 16,
    band: 9,
    blocks: 3,
  },
  {
    name: "vec4 H2 T136 D16 band 20 / 5 row blocks",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 136,
    depth: 16,
    band: 20,
    blocks: 5,
  },
  {
    name: "s16 vec4 H2 T40 D16 band 7",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 40,
    depth: 16,
    band: 7,
    blocks: 1,
    s16: true,
  },
  {
    name: "s16 vec4 GQA H6:2 T136 D16 band 9 / 3 row blocks",
    batch: 1,
    heads: 6,
    kvHeads: 2,
    tokens: 136,
    depth: 16,
    band: 9,
    blocks: 3,
    s16: true,
  },
  {
    // Rows m > N − 1 + band = 26 see no column; they fall in the second row block.
    name: "scalar GQA H4:2 M45 N21 D12 band 6 (rows 27+ empty) / 2 row blocks",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    tokens: 45,
    keys: 21,
    depth: 12,
    band: 6,
    blocks: 2,
  },
  {
    name: "vec4 H2 M24 N40 D16 band 5",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    tokens: 24,
    keys: 40,
    depth: 16,
    band: 5,
    blocks: 1,
  },
];

const PART = "attention-band";

/** The cases of {@link checkAttentionBand}: one per entry of {@link ATTENTION_BAND_GRID}. */
export const ATTENTION_BAND_CASES = {
  [PART]: ATTENTION_BAND_GRID.length,
} as const satisfies ExpectedCases;

/** The f32 half scale `D^-¼` (applied to both q and k). */
const halfScale = (depth: number): number => Math.fround(depth ** -0.25);

/** The node's attrs: the half scale, and `band` on the band form (`undefined` = the mask form). */
export const attentionBandAttrs = (
  testCase: AttentionBandCase,
  band: number | undefined,
): Readonly<Record<string, number>> =>
  band === undefined
    ? { scale: halfScale(testCase.depth) }
    : { scale: halfScale(testCase.depth), band };

/**
 * Deterministic inputs (no random numbers — a failure would not reproduce). Every fifth q row and
 * every seventh k row is −0 in every component: their scores are zeros of either sign, depending on
 * how the backend sums `+0 + (−0)·x`, which is where the band's select and the mask's add differ.
 */
const operand = (
  shape: readonly number[],
  zeroEvery: number,
  value: (index: number) => number,
): FilledTensor => {
  const [, , rows, depth] = shape;
  const count = shape.reduce((total, dim) => total * dim, 1);
  const data = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const row = Math.floor(index / depth) % rows;
    data[index] = row % zeroEvery === zeroEvery - 1 ? -0 : value(index);
  }
  return { dtype: "f32", shape, data };
};

const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** The additive `[1,1,M,N]` mask of the band: 0 where `|m − n| ≤ band`, −inf elsewhere. */
export const attentionBandMask = (rows: number, cols: number, band: number): FilledTensor => {
  const data = new Float32Array(rows * cols);
  for (let m = 0; m < rows; m += 1) {
    for (let n = 0; n < cols; n += 1) {
      data[m * cols + n] = Math.abs(m - n) <= band ? 0 : Number.NEGATIVE_INFINITY;
    }
  }
  return { dtype: "f32", shape: [1, 1, rows, cols], data };
};

/**
 * One attention node: with `band`, the band form (3 inputs and `attrs.band`); without, the mask
 * form (the mask a fourth graph input).
 */
export const attentionBandGraph = (
  testCase: AttentionBandCase,
  band: number | undefined,
): DeclarationJson => {
  const { batch, heads, kvHeads, tokens, depth } = testCase;
  const keys = attentionBandKeys(testCase);
  const q = [batch, heads, tokens, depth];
  const kv = [batch, kvHeads, keys, depth];
  const masked = band === undefined;
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention"] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: q },
      { name: "k", dtype: "f32", shape: kv },
      { name: "v", dtype: "f32", shape: kv },
      ...(masked ? [{ name: "mask", dtype: "f32", shape: [1, 1, tokens, keys] }] : []),
    ],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: q } },
    nodes: [
      {
        op: "attention",
        ins: masked ? ["q", "k", "v", "mask"] : ["q", "k", "v"],
        outs: ["o"],
        attrs: attentionBandAttrs(testCase, band),
      },
    ],
  };
};

/** The feeds of a case (the mask plan also takes `mask`). */
export const attentionBandFeeds = (
  testCase: AttentionBandCase,
): Readonly<Record<string, FilledTensor>> => {
  const { batch, heads, kvHeads, tokens, depth } = testCase;
  const keys = attentionBandKeys(testCase);
  return {
    q: operand([batch, heads, tokens, depth], 5, QUERY),
    k: operand([batch, kvHeads, keys, depth], 7, KEY),
    v: fill([batch, kvHeads, keys, depth], VALUE),
    mask: attentionBandMask(tokens, keys, testCase.band),
  };
};

/**
 * The Session options of a case: the budget that splits S into `blocks` row blocks (the largest
 * block holds ⌈M / blocks⌉ rows of `B·H · N` scores — the split rule of `planRowBlocks`), and the
 * f16 storage or compute it names.
 */
export const attentionBandOptions = (testCase: AttentionBandCase): SessionOptions => {
  const { batch, heads, tokens, blocks } = testCase;
  const keys = attentionBandKeys(testCase);
  const scoreBytes = testCase.s16 === true || testCase.c16 === true ? 2 : 4;
  return {
    ...(blocks === 1
      ? {}
      : {
          attentionRowBlockMaxBytes:
            batch * heads * Math.max(keys * scoreBytes, 8) * Math.ceil(tokens / blocks),
        }),
    ...(testCase.s16 === true ? { attentionScoreStorage: "f16" } : {}),
    ...(testCase.c16 === true ? { attentionCompute: "f16" } : {}),
  };
};

/**
 * The ①QK dispatches of a side's last run whose key carries `word`, the row-window word `:rwa`
 * exactly when the case splits, `:s16` exactly when it stores S in f16 and `:c16` exactly when it
 * computes in f16 (a silent fallback to another form counts as no dispatch).
 */
const qkDispatches = (session: Session, word: string, testCase: AttentionBandCase): number =>
  (session.diagnostics().lastRunPipelines ?? [])
    .filter(
      ({ key }) =>
        key.startsWith("attention_qk:") &&
        key.includes(word) &&
        key.includes(":rwa") === testCase.blocks > 1 &&
        key.includes(":s16") === (testCase.s16 === true) &&
        key.includes(":c16") === (testCase.c16 === true),
    )
    .reduce((total, { dispatchCount }) => total + dispatchCount, 0);

/**
 * The output words of the rows that see no key column under `band` (`m > N − 1 + band`, only when
 * M > N) that are not 0 — the empty row's output is 0 (ADR 0044's addendum).
 */
const emptyRowNonzeros = (
  testCase: AttentionBandCase,
  band: number,
  output: ArrayLike<number>,
): number => {
  const { batch, heads, tokens, depth } = testCase;
  const firstEmpty = attentionBandKeys(testCase) + band;
  let nonzeros = 0;
  for (let head = 0; head < batch * heads; head += 1) {
    for (let row = firstEmpty; row < tokens; row += 1) {
      for (let d = 0; d < depth; d += 1) {
        if (output[(head * tokens + row) * depth + d] !== 0) nonzeros += 1;
      }
    }
  }
  return nonzeros;
};

/**
 * Runs one case: the band Session and the mask Session on the same feeds (each under the error
 * scopes), O compared word for word (the mask side's words are the expected ones), and each side's
 * ①QK dispatches by key. `bandOffset` declares `band + bandOffset` on the band side only (the fault
 * injection of the node test; 0 in the check).
 */
export const runAttentionBandCase = async (
  gpu: GpuContext,
  testCase: AttentionBandCase,
  bandOffset = 0,
): Promise<CheckCase> => {
  const where = `${PART} ${testCase.name}`;
  const options = attentionBandOptions(testCase);
  const declared = testCase.band + bandOffset;
  const banded = await openTwinSession(
    gpu,
    `${where} (band)`,
    await openGraphModel(attentionBandGraph(testCase, declared)),
    options,
  );
  try {
    const masked = await openTwinSession(
      gpu,
      `${where} (mask)`,
      await openGraphModel(attentionBandGraph(testCase, undefined)),
      options,
    );
    try {
      const { mask, ...feeds } = attentionBandFeeds(testCase);
      const bandOut = await scopedGpuWork(gpu, `${where}: band run`, () => banded.run(feeds));
      const maskOut = await scopedGpuWork(gpu, `${where}: mask run`, () =>
        masked.run({ ...feeds, mask }),
      );
      const empty = testCase.tokens > attentionBandKeys(testCase) + declared;
      return {
        label: testCase.name,
        criteria: [
          compareWords("output o", wordsOf(bandOut["o"].data), wordsOf(maskOut["o"].data)),
          compareCount(
            "band side ①QK :band dispatches",
            qkDispatches(banded, ":band", testCase),
            testCase.blocks,
          ),
          compareCount(
            "mask side ①QK :mask dispatches",
            qkDispatches(masked, ":mask", testCase),
            testCase.blocks,
          ),
          ...(empty
            ? [
                compareCount(
                  "band side nonzero words of the rows with no visible column",
                  emptyRowNonzeros(testCase, declared, bandOut["o"].data),
                  0,
                ),
              ]
            : []),
        ],
      };
    } finally {
      await masked.dispose();
    }
  } finally {
    await banded.dispose();
  }
};

/** Every case of {@link ATTENTION_BAND_GRID} (one part). */
export const checkAttentionBand = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const cases: CheckCase[] = [];
  for (const testCase of ATTENTION_BAND_GRID) {
    throwIfAborted(options);
    cases.push(await runAttentionBandCase(gpu, testCase));
    options.onProgress?.({ part: PART, done: cases.length, total: ATTENTION_BAND_GRID.length });
  }
  return {
    id: PART,
    title: "Attention band vs the additive band mask (u32 identity of O)",
    cases,
    requirements: [],
  };
};
