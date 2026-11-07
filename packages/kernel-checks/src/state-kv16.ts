/**
 * Check 6 — f16 storage of a full state slot (ADR 0143). The contract is one rounding point: every
 * reader over a packed slot writes the same words as its f32 form over the same slot rounded on the
 * host, and `state_append` packs what the host's rounding packs, a word at a time, leaving every other
 * word alone.
 *
 * The readers rest on the backend compiling the two forms alike and on `unpack2x16float` being exact,
 * subnormals included, which is why one family of cases holds a slot of tiny values. The append rounds
 * in integers, and one case runs it on every rounding boundary of f16. WGSL guarantees none of the
 * backend's part, so this is the check to run on each backend.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { stateSliding } from "../../runtime/src/kernels/state-attention.ts";
import {
  runStateAppendF16,
  runStateAttentionReadonly,
  runStatePv,
  runStateQk,
  type StateAppendCase,
  type StateCase,
  type StateInputs,
  type StateMutation,
  type StatePipelineCache,
  type StatePvVariant,
  type StateQkVariant,
} from "../../runtime/tests/helpers/state-dispatch.ts";
import {
  E2B_FULL,
  E4B_FULL,
  emptyCensus,
  censusText,
  inputsOf,
  MHA_FULL,
  MINICPM5_FULL,
  type Shape,
  tally,
} from "./state-attention-rewrites.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareWords,
  type Criterion,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

/** 12B's full layers (one kv plane per batch row, r = 16), which no other check runs. */
const TWELVE_B_FULL: Shape = { name: "12b-full", heads: 16, kvHeads: 1, depth: 512, window: 0 };

const SHAPES: readonly Shape[] = [E2B_FULL, E4B_FULL, TWELVE_B_FULL, MINICPM5_FULL, MHA_FULL];
const PASTS: readonly number[] = [0, 17, 128, 1000];
/** (M, Q, rowsBlock): decode, verify with pad rows, a split block, and prefill (the tiled forms). */
const ROWS: readonly (readonly [number, number, number])[] = [
  [1, 1, 1],
  [4, 2, 4],
  [8, 5, 8],
  [8, 3, 4],
  [16, 16, 16],
  [32, 20, 32],
];

/** Most of a slot's values, rounded, are subnormal in f16. */
const TINY_SCALE = 2 ** -17;

/** The folded ①' exists for r = 4 and 8 at M ≤ 8. */
const foldable = (spec: StateCase): boolean => {
  const repeat = spec.heads / spec.kvHeads;
  return (repeat === 4 || repeat === 8) && spec.chunkRows <= 8;
};

const qkVariantsOf = (spec: StateCase): readonly StateQkVariant[] => [
  "sequential",
  ...(spec.chunkRows <= 8 ? (["parallel"] as const) : (["tiled"] as const)),
  ...(foldable(spec) ? (["folded"] as const) : []),
];

const pvVariantsOf = (spec: StateCase): readonly StatePvVariant[] => [
  "sequential",
  ...(spec.chunkRows < 16 ? (["parallel"] as const) : (["tiled"] as const)),
];

type GridCase = {
  readonly spec: StateCase;
  readonly nonFinite: boolean;
  readonly tiny: boolean;
  readonly seed: number;
};

const specOf = (shape: Shape, past: number, rows: readonly [number, number, number]): StateCase => {
  const [chunkRows, query, rowsBlock] = rows;
  return {
    name: `${shape.name} P=${past} M=${chunkRows} Q=${query} block=${rowsBlock}`,
    batch: 2,
    heads: shape.heads,
    kvHeads: shape.kvHeads,
    chunkRows,
    depth: shape.depth,
    capacity: past + chunkRows + 5,
    window: shape.window,
    past,
    query,
    rowsBlock,
  };
};

const grid = (): readonly GridCase[] =>
  [
    ...SHAPES.flatMap((shape) =>
      PASTS.flatMap((past) => ROWS.map((rows) => specOf(shape, past, rows))),
    ).flatMap((spec) => [false, true].map((nonFinite) => ({ spec, nonFinite, tiny: false }))),
    ...ROWS.map((rows) => ({
      spec: specOf(TWELVE_B_FULL, 17, rows),
      nonFinite: false,
      tiny: true,
    })),
  ].map((entry, index) => ({ ...entry, seed: 700_000 + index }));

const tinyInputs = (inputs: StateInputs): StateInputs => ({
  ...inputs,
  slotK: inputs.slotK.map((value) => value * TINY_SCALE),
  slotV: inputs.slotV.map((value) => value * TINY_SCALE),
});

/** The slot rounded on the host (the f32 form's input; the harness packs the same rounding). */
const rounded = (slot: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(slot.length);
  out.set(new Float16Array(slot));
  return out;
};

const roundedInputs = (inputs: StateInputs): StateInputs => ({
  ...inputs,
  slotK: rounded(inputs.slotK),
  slotV: rounded(inputs.slotV),
});

const wordsOfFloats = (data: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const checkReaders = async (
  gpu: GpuContext,
  options: CheckOptions,
  cache: StatePipelineCache,
): Promise<CheckPart> => {
  const census = emptyCensus();
  const cases: CheckCase[] = [];
  const all = grid();
  for (const { spec, nonFinite, tiny, seed } of all) {
    throwIfAborted(options);
    const label = `${spec.name}${nonFinite ? " non-finite" : ""}${tiny ? " tiny" : ""}`;
    const generated = inputsOf(spec, seed, nonFinite);
    const inputs = tiny ? tinyInputs(generated) : generated;
    const reference = roundedInputs(inputs);
    const criteria = await scopedGpuWork(gpu, label, async () => {
      const found: Criterion[] = [];
      for (const variant of qkVariantsOf(spec)) {
        const expected = await runStateQk(gpu.device, spec, reference, variant, { cache });
        const actual = await runStateQk(gpu.device, spec, inputs, variant, {
          cache,
          storage: "f16",
        });
        expected.forEach((words, block) => {
          found.push(
            compareWords(
              `① ${variant}, S words of block ${block}: f16 vs f32 over a rounded slot`,
              actual[block] ?? new Uint32Array(0),
              words,
            ),
          );
        });
      }
      for (const variant of pvVariantsOf(spec)) {
        const expected = await runStatePv(gpu.device, spec, reference, variant, { cache });
        const actual = await runStatePv(gpu.device, spec, inputs, variant, {
          cache,
          storage: "f16",
        });
        found.push(
          compareWords(`③ ${variant}, O words: f16 vs f32 over a rounded slot`, actual, expected),
        );
        tally(census, expected);
      }
      if (spec.chunkRows === 1 && spec.query === 1) {
        const expected = await runStateAttentionReadonly(gpu.device, spec, reference, { cache });
        const actual = await runStateAttentionReadonly(gpu.device, spec, inputs, {
          cache,
          storage: "f16",
        });
        found.push(
          compareWords(
            "readonly, S words: f16 vs f32 over a rounded slot",
            wordsOfFloats(actual.scores),
            wordsOfFloats(expected.scores),
          ),
          compareWords(
            "readonly, O words: f16 vs f32 over a rounded slot",
            wordsOfFloats(actual.out),
            wordsOfFloats(expected.out),
          ),
        );
      }
      return found;
    });
    cases.push({ label, criteria });
    options.onProgress?.({ part: "state-kv16-readers", done: cases.length, total: all.length });
  }
  return {
    id: "state-kv16-readers",
    title: "f16 full-slot readers vs their f32 forms over a rounded slot",
    cases,
    requirements: [
      {
        id: "census",
        name: "the reference O met finite and NaN words",
        met: census.finite > 0 && census.nan > 0,
        detail: censusText(census),
      },
    ],
  };
};

/** Values around f16's rounding boundaries, inside its finite normal range (the common cases). */
const APPEND_VALUES: readonly number[] = [
  1,
  -1,
  -0,
  1 + 2 ** -11, // tie → even side 1.0
  1 + 3 * 2 ** -11, // tie → even side 1 + 2^-9
  1 + 2 ** -11 + 2 ** -20, // above the tie
  -(1 + 2 ** -11),
  2 ** -14, // the smallest normal
  65504, // the largest finite
  65519, // below the overflow tie
  0.1,
  22.143,
  3.3,
];

const APPEND_POISON = 0xdeadbeef;

const APPEND_SHAPES: readonly Pick<StateAppendCase, "depth" | "kvPlanes">[] = [
  { depth: 8, kvPlanes: 2 },
  { depth: 512, kvPlanes: 2 },
];
/** (P, Q, M): decode, pad rows left out, a whole chunk. */
const APPEND_LENGTHS: readonly (readonly [number, number, number])[] = [
  [0, 1, 1],
  [3, 2, 4],
  [5, 4, 4],
];

/** The next f32 above (`step` 1) or below (`step` −1) a positive value. */
const f32Neighbour = (value: number, step: 1 | -1): number => {
  const word = new Float32Array([value]);
  new Uint32Array(word.buffer)[0] += step;
  return word[0];
};

/**
 * Every f16 value and, for each pair of neighbours, the midpoint and the f32 values on either side of
 * it, in both signs: every rounding decision the append can take, overflow and subnormals included.
 */
const ROUNDING_BOUNDARIES: readonly number[] = (() => {
  const halves = new Uint16Array(2);
  const values = new Float16Array(halves.buffer);
  const positive: number[] = [65536, 1e30, Infinity, 1e-40, 2 ** -149, 2 ** -26];
  for (let bits = 0; bits < 0x7c00; bits += 1) {
    halves[0] = bits;
    halves[1] = bits + 1;
    const above = bits === 0x7bff ? 65536 : values[1];
    const midpoint = (values[0] + above) / 2;
    positive.push(values[0], f32Neighbour(midpoint, -1), midpoint, f32Neighbour(midpoint, 1));
  }
  return [...positive.flatMap((value) => [value, -value]), NaN];
})();

const BOUNDARY_ROWS = Math.ceil(ROUNDING_BOUNDARIES.length / (2 * 512));

const appendSpecs = (): readonly StateAppendCase[] => [
  ...APPEND_SHAPES.flatMap(({ depth, kvPlanes }) =>
    APPEND_LENGTHS.map(([past, query, chunkRows]) => ({
      kvPlanes,
      chunkRows,
      depth,
      capacity: past + query + 3,
      window: 0,
      past,
      query,
    })),
  ),
  {
    kvPlanes: 2,
    chunkRows: BOUNDARY_ROWS,
    depth: 512,
    capacity: BOUNDARY_ROWS,
    window: 0,
    past: 0,
    query: BOUNDARY_ROWS,
  },
];

const appendInput = (spec: StateAppendCase): Float32Array<ArrayBuffer> => {
  const x = new Float32Array(spec.kvPlanes * spec.chunkRows * spec.depth);
  const values = spec.chunkRows === BOUNDARY_ROWS ? ROUNDING_BOUNDARIES : APPEND_VALUES;
  for (let i = 0; i < x.length; i += 1) x[i] = values[i % values.length];
  return x;
};

const appendWords = (spec: StateAppendCase): number =>
  (spec.kvPlanes * spec.capacity * spec.depth) / 2;

/** What the host's rounding packs, word for word; the poison everywhere else. */
const expectedSlot = (
  spec: StateAppendCase,
  x: Float32Array<ArrayBuffer>,
): Uint32Array<ArrayBuffer> => {
  const slot = new Uint32Array(appendWords(spec)).fill(APPEND_POISON);
  const pair = new Float16Array(2);
  const word = new Uint32Array(pair.buffer);
  for (let plane = 0; plane < spec.kvPlanes; plane += 1) {
    for (let row = 0; row < spec.query; row += 1) {
      for (let w = 0; w < spec.depth / 2; w += 1) {
        const src = (plane * spec.chunkRows + row) * spec.depth + w * 2;
        pair[0] = x[src];
        pair[1] = x[src + 1];
        slot[((plane * spec.capacity + spec.past + row) * spec.depth) / 2 + w] = word[0];
      }
    }
  }
  return slot;
};

const appendCase = (spec: StateAppendCase): string =>
  `D=${spec.depth} P=${spec.past} Q=${spec.query} M=${spec.chunkRows}`;

const checkAppend = async (
  gpu: GpuContext,
  options: CheckOptions,
  cache: StatePipelineCache,
): Promise<CheckPart> => {
  const cases: CheckCase[] = [];
  const specs = appendSpecs();
  for (const spec of specs) {
    throwIfAborted(options);
    const label = appendCase(spec);
    const x = appendInput(spec);
    const actual = await scopedGpuWork(gpu, label, () =>
      runStateAppendF16(gpu.device, spec, x, appendWords(spec), APPEND_POISON, { cache }),
    );
    cases.push({
      label,
      criteria: [compareWords("slot words vs the host's rounding", actual, expectedSlot(spec, x))],
    });
    options.onProgress?.({ part: "state-kv16-append", done: cases.length, total: specs.length });
  }
  return {
    id: "state-kv16-append",
    title: "f16 state_append vs the host's rounding (slot words)",
    cases,
    requirements: [],
  };
};

export const STATE_KV16_CASES = {
  "state-kv16-readers": SHAPES.length * PASTS.length * ROWS.length * 2 + ROWS.length,
  "state-kv16-append": APPEND_SHAPES.length * APPEND_LENGTHS.length + 1,
};

/** A mutant of the check: the words of its output that differ from the correct form's. */
export type StateKv16Mutant = {
  readonly label: string;
  readonly differingWords: number;
  readonly words: number;
};

const replaced = (wgsl: string, from: string, to: string, label: string): string => {
  if (!wgsl.includes(from)) {
    throw new Error(`mutant '${label}': the target is not in the WGSL (${from})`);
  }
  return wgsl.replaceAll(from, to);
};

/** Mutates only the kernels that hold the f16 reader. */
const readerMutant =
  (from: string, to: string, label: string): StateMutation =>
  (kernel, wgsl) =>
    kernel === "stats" ? wgsl : replaced(wgsl, from, to, label);

const READER_MUTANTS: readonly { readonly label: string; readonly mutate: StateMutation }[] = [
  {
    label: "the reader swaps the two halves of a word",
    mutate: readerMutant("select(pair.x, pair.y,", "select(pair.y, pair.x,", "swapped halves"),
  },
  {
    label: "the reader picks the half by the wrong parity",
    mutate: readerMutant("(i & 1u) == 1u", "(i & 1u) == 0u", "wrong parity"),
  },
  {
    label: "the reader reads the neighbouring word",
    mutate: readerMutant("[i >> 1u]", "[(i >> 1u) ^ 1u]", "neighbouring word"),
  },
];

const APPEND_MUTANTS: readonly { readonly label: string; readonly mutate: StateMutation }[] = [
  {
    label: "the append swaps the two halves of a word",
    mutate: (_kernel, wgsl) =>
      replaced(
        wgsl,
        "f16_bits(x[src]) | (f16_bits(x[src + 1u]) << 16u)",
        "f16_bits(x[src + 1u]) | (f16_bits(x[src]) << 16u)",
        "append swap",
      ),
  },
  {
    label: "the append rounds toward zero, as pack2x16float does on D3D12",
    mutate: (_kernel, wgsl) =>
      replaced(
        wgsl,
        "u32(rest > half || (rest == half && (kept & 1u) == 1u))",
        "0u",
        "append toward zero",
      ),
  },
  {
    label: "the append writes the neighbouring word",
    mutate: (_kernel, wgsl) =>
      replaced(wgsl, "* words + w;", "* words + (w ^ 1u);", "append neighbour"),
  },
];

const differing = (
  a: Uint32Array<ArrayBuffer>,
  b: Uint32Array<ArrayBuffer>,
): { readonly differingWords: number; readonly words: number } => ({
  differingWords: compareWords("mutant", a, b).failures,
  words: b.length,
});

/**
 * Each mutant must change the output on a case the identity passes: a zero-mismatch result proves
 * nothing otherwise. Readers run on ③ over the E2B shape with a past long enough to read the slot.
 */
export const checkStateKv16Mutants = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<readonly StateKv16Mutant[]> => {
  const spec = specOf(E2B_FULL, 128, [1, 1, 1]);
  const inputs = inputsOf(spec, 700_000, false);
  const results: StateKv16Mutant[] = [];
  const expected = await runStatePv(gpu.device, spec, inputs, "parallel", { storage: "f16" });
  for (const { label, mutate } of READER_MUTANTS) {
    throwIfAborted(options);
    const actual = await runStatePv(gpu.device, spec, inputs, "parallel", {
      storage: "f16",
      mutate,
    });
    results.push({ label, ...differing(actual, expected) });
  }
  const appendSpec = appendSpecs()[1];
  const x = appendInput(appendSpec);
  const slot = expectedSlot(appendSpec, x);
  for (const { label, mutate } of APPEND_MUTANTS) {
    throwIfAborted(options);
    const actual = await runStateAppendF16(
      gpu.device,
      appendSpec,
      x,
      appendWords(appendSpec),
      APPEND_POISON,
      { mutate },
    );
    results.push({ label, ...differing(actual, slot) });
  }
  return results;
};

export const STATE_KV16_MUTANTS = READER_MUTANTS.length + APPEND_MUTANTS.length;

export const checkStateKv16 = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<readonly CheckPart[]> => {
  if (SHAPES.some((shape) => stateSliding(shape.window))) {
    throw new Error("f16 KV storage is for full slots only");
  }
  const cache: StatePipelineCache = new Map();
  return [await checkReaders(gpu, options, cache), await checkAppend(gpu, options, cache)];
};
