// Behaviour tests of the draw from the leading entries of the raw logits' ranking
// (`Sampler.topCandidates` / `candidateDistribution` in `src/generation/sampler.ts`). Neither a GPU
// nor real assets are needed: the GPU `topk`'s ranking is modelled on the host (`leading`).
//
// What this pins is **identity with the whole-row draw**: for the same spec, seed and row the
// candidates give the same draw target bit for bit — the same tokens in the same order with the
// same f64 probabilities — and so the same token, with the RNG consumed alike; or they say they do
// not decide, and then the whole row is what draws. A difference here raises no exception on a real
// GPU: it shows only as another token somewhere in a text.

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  candidateDistribution,
  createSampler,
  samplerDistribution,
  type SamplerSpec,
} from "../src/generation/sampler.ts";

/** The largest selection the tests let the capability ask for (a device with 32,768 B of workgroup storage). */
const MAX_COUNT = 127;

/**
 * The first `count` entries of `row` in the GPU `topk`'s order — value descending, the smaller id
 * first on a tie, +0 and −0 equal, NaN above every number — as the selection returns them (the raw
 * value bits, the ids).
 *
 * Written apart from the implementation's rule: a rank per id (2 for NaN, 1 for a number), then the
 * number, then the id.
 */
const leading = (
  row: Float32Array<ArrayBuffer>,
  count: number,
): { readonly values: Float32Array<ArrayBuffer>; readonly indices: Int32Array<ArrayBuffer> } => {
  const ids = Array.from({ length: row.length }, (_unused, id) => id);
  ids.sort((left, right) => {
    const leftNaN = Number.isNaN(row[left]);
    const rightNaN = Number.isNaN(row[right]);
    if (leftNaN !== rightNaN) return leftNaN ? -1 : 1;
    if (!leftNaN && row[left] !== row[right]) return row[right] - row[left] > 0 ? 1 : -1;
    return left - right;
  });
  const indices = Int32Array.from(ids.slice(0, count));
  return { values: Float32Array.from(indices, (id) => row[id]), indices };
};

/** A deterministic uniform stream for building rows (not the sampler's RNG). */
const stream = (seed: number): (() => number) => {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return (state + 0.5) / 4294967296;
  };
};

/** A row of `size` logits: a Gaussian bulk with a few high entries, as a decode row has. */
const decodeRow = (size: number, uniform: () => number): Float32Array<ArrayBuffer> => {
  const row = new Float32Array(size);
  for (let id = 0; id < size; id += 1) {
    row[id] = Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()) * 3 - 5;
  }
  for (let high = 0; high < 12; high += 1) row[Math.floor(uniform() * size)] = 8 + uniform() * 6;
  return row;
};

/** A row of `size` logits at `fill`, with `entries` set. */
const rowOf = (
  size: number,
  fill: number,
  entries: readonly (readonly [id: number, value: number])[],
): Float32Array<ArrayBuffer> => {
  const row = new Float32Array(size).fill(fill);
  for (const [id, value] of entries) row[id] = value;
  return row;
};

/** The f32 just below `value`. */
const below = (value: number): number => {
  const cell = new Float32Array([value]);
  const bits = new Uint32Array(cell.buffer);
  bits[0] += value > 0 ? -1 : 1;
  return cell[0];
};

/**
 * The capability's draw target for `row` (`undefined` when the entries do not decide it), with the
 * count the capability asks for.
 */
const fromCandidates = (
  row: Float32Array<ArrayBuffer>,
  spec: SamplerSpec,
  suppressed: readonly number[] = [],
) => {
  const topK = Math.min(spec.topK ?? 0, row.length);
  const { values, indices } = leading(row, topK + suppressed.length + 1);
  return candidateDistribution(values, indices, spec, suppressed, row.length, topK);
};

describe("the draw from the leading entries of the raw logits", () => {
  describe("which specs get it", () => {
    it("is offered for a top-k draw without a repetition penalty or a logit bias, asking for top-k + the suppressed tokens + 1 entries", () => {
      const spec: SamplerSpec = { temperature: 1, topK: 64, topP: 0.95, seed: 0 };
      assertEquals(createSampler(spec).topCandidates(262144, MAX_COUNT)?.count, 65);
      assertEquals(
        createSampler(spec, [258882, 258883]).topCandidates(262144, MAX_COUNT)?.count,
        67,
      );
      assertEquals(
        createSampler({ temperature: 0.7, topK: 5 }).topCandidates(262144, MAX_COUNT)?.count,
        6,
      );
    });

    it("is not offered for a spec whose draw the raw ranking does not carry", () => {
      for (const [name, spec] of [
        ["temperature 0", { topK: 64 }],
        ["no top-k", { temperature: 1, topP: 0.95 }],
        ["a repetition penalty", { temperature: 1, topK: 64, repetitionPenalty: 1.1 }],
        ["a logit bias", { temperature: 1, topK: 64, logitBias: [[7, -1]] }],
      ] as const) {
        assertEquals(createSampler(spec).topCandidates(262144, MAX_COUNT), undefined, name);
      }
    });

    it("is not offered when it needs more entries than the selection returns or the row has", () => {
      const spec: SamplerSpec = { temperature: 1, topK: 64 };
      // A device at WebGPU's default workgroup storage selects at most 63.
      assertEquals(createSampler(spec).topCandidates(262144, 63), undefined);
      assertEquals(createSampler(spec).topCandidates(262144, 65)?.count, 65);
      // top-k is clamped to the vocabulary, and the extra entry does not exist.
      assertEquals(createSampler(spec).topCandidates(16, MAX_COUNT), undefined);
      assertEquals(
        createSampler({ temperature: 1, topK: 15 }).topCandidates(16, MAX_COUNT)?.count,
        16,
      );
    });
  });

  describe("on ordinary rows", () => {
    it("gives the whole row's draw target bit for bit, over temperatures, top-k, top-p and suppressed sets", () => {
      const uniform = stream(20261118);
      let compared = 0;
      for (const temperature of [0.3, 0.7, 1, 1.5, 4]) {
        for (const topK of [1, 2, 5, 40, 64]) {
          for (const topP of [undefined, 0.3, 0.95, 1]) {
            for (const suppressed of [[], [3, 4000]]) {
              const spec: SamplerSpec = {
                temperature,
                topK,
                ...(topP === undefined ? {} : { topP }),
              };
              const row = decodeRow(4096, uniform);
              // The suppressed ids rank high on some rows.
              if (suppressed.length > 0 && uniform() < 0.5) row[4000] = 40;
              const expected = samplerDistribution(row, spec, [], suppressed);
              const actual = fromCandidates(row, spec, suppressed);
              assert(actual !== undefined, `not decided: ${JSON.stringify(spec)}`);
              assertEquals([...actual.tokens], [...expected.tokens], JSON.stringify(spec));
              // Float64Array equality is bit equality here (no NaN, and ±0 does not occur).
              assertEquals(actual.probabilities, expected.probabilities, JSON.stringify(spec));
              compared += 1;
            }
          }
        }
      }
      assertEquals(compared, 200);
    });

    it("picks the token the whole row picks at every step of a generation, for the same seed", () => {
      for (const seed of [0, 1, 20261118]) {
        for (const spec of [
          { temperature: 1, topK: 64, topP: 0.95, seed },
          { temperature: 0.7, topK: 40, seed },
          { temperature: 2.5, topK: 3, topP: 0.5, seed },
        ] satisfies SamplerSpec[]) {
          const uniform = stream(seed + 7);
          const whole = createSampler(spec);
          const candidates = createSampler(spec).topCandidates(4096, MAX_COUNT);
          assert(candidates !== undefined);
          for (let step = 0; step < 64; step += 1) {
            const row = decodeRow(4096, uniform);
            const { values, indices } = leading(row, candidates.count);
            assertEquals(candidates.next(values, indices), whole.next(row, []), `step ${step}`);
          }
        }
      }
    });

    it("advances the RNG only when the whole-row draw would: a single candidate draws nothing", () => {
      const uniform = stream(5);
      // top-p keeps one candidate on a row with one dominant entry, several on a flat one.
      const spec: SamplerSpec = { temperature: 1, topK: 8, topP: 0.9, seed: 3 };
      const whole = createSampler(spec);
      const mixed = createSampler(spec);
      const candidates = mixed.topCandidates(512, MAX_COUNT);
      assert(candidates !== undefined);
      for (let step = 0; step < 24; step += 1) {
        const row =
          step % 2 === 0
            ? rowOf(512, -5, [[step, 30]])
            : decodeRow(512, uniform).map((value) => value / 16);
        const { values, indices } = leading(row, candidates.count);
        assertEquals(candidates.next(values, indices), whole.next(row, []), `step ${step}`);
      }
      // The two generators are at the same place: a whole-row draw from each agrees.
      const row = decodeRow(512, uniform).map((value) => value / 16);
      assertEquals(mixed.next(row, []), whole.next(row, []));
    });
  });

  describe("on ties", () => {
    it("keeps the smaller id at the top-k boundary, as the whole row does, without giving up at temperature 1", () => {
      // Five entries at the same logit across the boundary of top-3; the candidates hold four.
      const row = rowOf(64, -5, [
        [40, 9],
        [50, 7],
        [12, 7],
        [33, 7],
        [20, 7],
        [61, 7],
      ]);
      const spec: SamplerSpec = { temperature: 1, topK: 3 };
      const expected = samplerDistribution(row, spec, []);
      assertEquals([...expected.tokens], [40, 12, 20]);
      const actual = fromCandidates(row, spec);
      assert(actual !== undefined, "a tie the entries decide was given up");
      assertEquals([...actual.tokens], [40, 12, 20]);
      assertEquals(actual.probabilities, expected.probabilities);
    });

    it("orders candidates that a temperature rounds to one value by id, as the whole row does", () => {
      // Two neighbouring f32 that 1 / 3 rounds to the same f32: the lower logit has the smaller id.
      const high = 1.75;
      const low = below(high);
      assertEquals(Math.fround(high / 3), Math.fround(low / 3));
      const row = rowOf(64, -5, [
        [30, high],
        [10, low],
        [50, 0.5],
        [5, 0.25],
      ]);
      const spec: SamplerSpec = { temperature: 3, topK: 3 };
      const expected = samplerDistribution(row, spec, []);
      // By the raw logits token 30 leads; by the processed ones the tie goes to the smaller id.
      assertEquals([...expected.tokens], [10, 30, 50]);
      const actual = fromCandidates(row, spec);
      assert(actual !== undefined);
      assertEquals([...actual.tokens], [10, 30, 50]);
      assertEquals(actual.probabilities, expected.probabilities);
    });

    it("treats +0 and −0 as one value", () => {
      const row = rowOf(32, -1, [
        [20, 0],
        [9, -0],
        [14, 0],
        [3, -0],
      ]);
      const spec: SamplerSpec = { temperature: 1, topK: 3 };
      const expected = samplerDistribution(row, spec, []);
      assertEquals([...expected.tokens], [3, 9, 14]);
      const actual = fromCandidates(row, spec);
      assertEquals([...(actual?.tokens ?? [])], [3, 9, 14]);
      assertEquals(actual?.probabilities, expected.probabilities);
    });

    it("does not decide when a tie under a temperature reaches past the last entry: a token outside them wins it by its id", () => {
      // Tokens 5 and 9 hold the higher logit, token 2 its lower neighbour; all three round to one
      // value, and top-1 of the processed row is token 2 — which the two leading entries lack.
      const high = 1.75;
      const low = below(high);
      const row = rowOf(64, -5, [
        [5, high],
        [9, high],
        [2, low],
      ]);
      const spec: SamplerSpec = { temperature: 3, topK: 1 };
      assertEquals([...samplerDistribution(row, spec, []).tokens], [2]);
      assertEquals([...leading(row, 2).indices], [5, 9]);
      assertEquals(fromCandidates(row, spec), undefined);
    });
  });

  describe("with suppressed tokens", () => {
    const SUPPRESSED = [50, 51] as const;

    it("leaves them out wherever they rank: above every candidate, on a tie with one, or nowhere near", () => {
      for (const [name, row] of [
        [
          "both above every candidate",
          rowOf(64, -5, [
            [50, 40],
            [51, 41],
            [7, 3],
            [9, 2],
            [30, 1],
          ]),
        ],
        [
          "one on a tie with a candidate",
          rowOf(64, -5, [
            [50, 3],
            [7, 3],
            [60, 3],
            [9, 2],
            [30, 1],
          ]),
        ],
        [
          "neither near the top",
          rowOf(64, -5, [
            [7, 3],
            [9, 2],
            [30, 1],
            [31, 0.5],
          ]),
        ],
      ] as const) {
        const spec: SamplerSpec = { temperature: 0.8, topK: 3, topP: 0.99 };
        const expected = samplerDistribution(row, spec, [], SUPPRESSED);
        const actual = fromCandidates(row, spec, SUPPRESSED);
        assert(actual !== undefined, name);
        assertEquals([...actual.tokens], [...expected.tokens], name);
        assertEquals(actual.probabilities, expected.probabilities, name);
        assert(!actual.tokens.includes(50) && !actual.tokens.includes(51), name);
      }
    });

    it("draws past a NaN at a suppressed id, which the whole row masks too", () => {
      const row = rowOf(64, -5, [
        [50, Number.NaN],
        [7, 3],
        [9, 2],
        [30, 1],
        [31, 0.5],
      ]);
      const spec: SamplerSpec = { temperature: 1, topK: 3 };
      const expected = samplerDistribution(row, spec, [], SUPPRESSED);
      const actual = fromCandidates(row, spec, SUPPRESSED);
      assertEquals([...(actual?.tokens ?? [])], [...expected.tokens]);
      assertEquals(actual?.probabilities, expected.probabilities);
    });
  });

  describe("when the entries do not decide the draw", () => {
    const spec: SamplerSpec = { temperature: 1, topK: 3, seed: 11 };

    /** `next` from `row`'s leading entries on a fresh sampler, and what the whole row then gives. */
    const undecided = (row: Float32Array<ArrayBuffer>, suppressed: readonly number[] = []) => {
      const sampler = createSampler(spec, suppressed);
      const candidates = sampler.topCandidates(row.length, MAX_COUNT);
      assert(candidates !== undefined);
      const { values, indices } = leading(row, candidates.count);
      return { drawn: candidates.next(values, indices), sampler };
    };

    it("says so for a NaN outside the suppressed set, and the whole row throws for the smallest such id", () => {
      const row = rowOf(64, -5, [
        [40, Number.NaN],
        [22, Number.NaN],
        [50, Number.NaN],
        [7, 3],
      ]);
      const { drawn, sampler } = undecided(row, [50]);
      assertEquals(drawn, undefined);
      assertThrows(() => sampler.next(row, []), Error, "logits[22] is NaN");
    });

    it("says so for a maximum that is not finite, and the whole row throws for it", () => {
      const row = rowOf(64, -5, [
        [7, Number.POSITIVE_INFINITY],
        [9, 2],
        [30, 1],
        [31, 0],
      ]);
      const { drawn, sampler } = undecided(row);
      assertEquals(drawn, undefined);
      assertThrows(() => sampler.next(row, []), Error, "non-finite (Infinity)");
    });

    it("says so when fewer than top-k candidates are above −∞", () => {
      const row = rowOf(64, Number.NEGATIVE_INFINITY, [
        [7, 3],
        [9, 2],
      ]);
      assertEquals(undecided(row).drawn, undefined);
    });

    it("leaves the RNG where it was, so the whole-row draw that follows is the one an untouched sampler makes", () => {
      const flat = decodeRow(64, stream(9)).map((value) => value / 16);
      const row = rowOf(64, -5, [
        [40, Number.NaN],
        [7, 3],
      ]);
      const { drawn, sampler } = undecided(row);
      assertEquals(drawn, undefined);
      assertEquals(sampler.next(flat, []), createSampler(spec).next(flat, []));
    });

    it("says so for entries that are not the ranking the selection promises, and never throws", () => {
      const row = rowOf(64, -5, [
        [40, 9],
        [12, 7],
        [33, 7],
        [20, 5],
        [61, 4],
      ]);
      const sampler = createSampler(spec);
      const candidates = sampler.topCandidates(64, MAX_COUNT);
      assert(candidates !== undefined);
      const good = leading(row, candidates.count);
      assertEquals([...good.indices], [40, 12, 33, 20]);
      const f32 = (values: readonly number[]) => Float32Array.from(values);
      const i32 = (values: readonly number[]) => Int32Array.from(values);
      for (const [name, values, indices] of [
        ["a tie with the larger id first", f32([9, 7, 7, 5]), i32([40, 33, 12, 20])],
        ["values ascending", f32([5, 7, 7, 9]), i32([20, 12, 33, 40])],
        ["an id repeated", f32([9, 7, 7, 5]), i32([40, 12, 12, 20])],
        ["an id outside the vocabulary", f32([9, 7, 7, 5]), i32([40, 12, 33, 64])],
        ["a negative id", f32([9, 7, 7, 5]), i32([40, 12, -1, 20])],
        ["a number above a NaN", f32([9, Number.NaN, 7, 5]), i32([40, 12, 33, 20])],
        ["too few entries", f32([9, 7, 7]), i32([40, 12, 33])],
        ["too many entries", f32([9, 7, 7, 5, 4]), i32([40, 12, 33, 20, 61])],
        ["no entry", f32([]), i32([])],
      ] as const) {
        assertEquals(candidates.next(values, indices), undefined, name);
      }
      // The same sampler still draws from entries that are the ranking.
      assertEquals(candidates.next(good.values, good.indices), createSampler(spec).next(row, []));
    });
  });
});
