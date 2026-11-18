/**
 * The candidate selection on a real GPU: the per-part `topk` at the real vocabulary, merged on the
 * host, gives the sampler **the draw target the whole row gives** — the same tokens in the same
 * order with the same probabilities — for ordinary rows, ties inside a part and across parts, ±0,
 * suppressed ids at the top, and a run of consecutive ids at the top; and it gives up, instead of
 * answering, for a NaN, for −∞ among the candidates and for a row whose leading entries crowd into
 * one part (one residue of the id modulo 256). The host half alone is `generation_leading_entries_test`
 * and `generation_sampler_candidates_test`; what only a GPU shows is that the kernel's own order is
 * the one they assume.
 */
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext, type Session } from "@karume/runtime";
import { gemmaCandidateLayout, gemmaCandidateSelectionModel } from "../src/gemma/greedy-output.ts";
import { leadingEntries } from "../src/generation/leading-entries.ts";
import {
  candidateDistribution,
  samplerDistribution,
  type SamplerSpec,
} from "../src/generation/sampler.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const VOCABULARY = 262144;
const SUPPRESSED = [258882, 258883] as const;
/** The distributions' sampler. */
const DECLARED: SamplerSpec = { temperature: 1, topK: 64, topP: 0.95 };

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

/** A decode-like row: a Gaussian bulk and a few dozen high entries anywhere in the vocabulary. */
const decodeRow = (seed: number): Float32Array<ArrayBuffer> => {
  const uniform = stream(seed);
  const row = new Float32Array(VOCABULARY);
  for (let id = 0; id < VOCABULARY; id += 1) {
    row[id] = Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()) * 3 - 5;
  }
  for (let high = 0; high < 48; high += 1) {
    row[Math.floor(uniform() * VOCABULARY)] = 9 + uniform() * 6;
  }
  return row;
};

const rowWith = (
  fill: number,
  entries: readonly (readonly [id: number, value: number])[],
): Float32Array<ArrayBuffer> => {
  const row = new Float32Array(VOCABULARY).fill(fill);
  for (const [id, value] of entries) row[id] = value;
  return row;
};

/** 70 distinct logits at ids spread over the vocabulary (no two in one part), highest first. */
const spread = (step: number): [number, number][] =>
  Array.from({ length: 70 }, (_unused, at) => [(at * step + 11) % VOCABULARY, 20 - at * 0.125]);

type Case = {
  readonly name: string;
  readonly row: Float32Array<ArrayBuffer>;
  readonly spec?: SamplerSpec;
  readonly suppressed?: readonly number[];
};

const DECIDED: readonly Case[] = [
  { name: "an ordinary row", row: decodeRow(1) },
  {
    name: "another ordinary row, at a temperature that is not 1",
    row: decodeRow(2),
    spec: { temperature: 0.7, topK: 64, topP: 0.95 },
  },
  {
    name: "an ordinary row, top-k 40 without top-p",
    row: decodeRow(3),
    spec: { temperature: 1.3, topK: 40 },
  },
  {
    name: "ties across the parts and inside one, at the top-k boundary",
    row: rowWith(-5, [
      ...spread(4099).slice(0, 62),
      // Four ids at one logit where two places are left: the two smallest ids are kept.
      [200_001, 3],
      [5, 3],
      [200_000, 3],
      [131_072, 3],
    ]),
  },
  {
    name: "+0 and −0 at the top",
    row: rowWith(-1, [
      [70_000, 0],
      [9, -0],
      [140_000, 0],
      [3, -0],
    ]),
    spec: { temperature: 1, topK: 3 },
  },
  {
    name: "the suppressed ids above every candidate",
    row: rowWith(-5, [...spread(4099), [258882, 40], [258883, 41]]),
    suppressed: SUPPRESSED,
  },
  {
    name: "a NaN at a suppressed id, which the draw masks",
    row: rowWith(-5, [...spread(4099), [258882, Number.NaN]]),
    suppressed: SUPPRESSED,
  },
  {
    name: "a run of 40 consecutive ids at the top (neighbours in the vocabulary are in different parts)",
    row: rowWith(-5, [
      ...spread(4099).slice(0, 30),
      ...Array.from({ length: 40 }, (_unused, at): [number, number] => [
        70_000 + at,
        30 - at * 0.125,
      ]),
    ]),
  },
  {
    name: "the last id of the vocabulary and the first, in the last and the first part",
    row: rowWith(-5, [...spread(4099).slice(0, 60), [262143, 30], [0, 29]]),
  },
];

const UNDECIDED: readonly Case[] = [
  {
    name: "a NaN outside the suppressed set",
    row: rowWith(-5, [...spread(4099), [77, Number.NaN]]),
  },
  {
    name: "the leading entries crowded into one part (20 of them at ids 256 apart)",
    row: rowWith(-5, [
      ...spread(4099).slice(0, 50),
      ...Array.from({ length: 20 }, (_unused, at): [number, number] => [
        1000 + at * 256,
        30 - at * 0.125,
      ]),
    ]),
  },
  {
    name: "fewer than top-k logits above −∞",
    row: rowWith(Number.NEGATIVE_INFINITY, spread(4099).slice(0, 20)),
  },
];

describe.skipIf(!GPU_AVAILABLE)("the candidate selection (real GPU)", () => {
  const layout = gemmaCandidateLayout(VOCABULARY);
  let gpu: GpuContext;
  let selector: Session;

  beforeAll(async () => {
    assert(layout !== undefined);
    gpu = await acquireGpu();
    selector = await gemmaCandidateSelectionModel(VOCABULARY, layout).createContainerSession(gpu);
  });

  afterAll(async () => {
    await selector.dispose();
    gpu.destroy();
  });

  /** The draw target from the GPU's selection of `row` (`undefined` = it does not decide). */
  const fromSelection = async ({ row, spec = DECLARED, suppressed = [] }: Case) => {
    assert(layout !== undefined);
    const out = await selector.run({
      logits: { dtype: "f32", shape: [1, 1, VOCABULARY], data: row },
    });
    const values = out.part_value;
    const indices = out.part_index;
    assert(values.dtype === "f32" && indices.dtype === "i32");
    const topK = spec.topK ?? 0;
    const entries = leadingEntries(
      { values: values.data, indices: indices.data, ...layout },
      topK + suppressed.length + 1,
    );
    return entries === undefined
      ? undefined
      : candidateDistribution(entries.values, entries.indices, spec, suppressed, VOCABULARY, topK);
  };

  it("cuts the Gemma 4 vocabulary into 256 parts of 1,024 ids and selects 8 entries of each", () => {
    assertEquals(layout, { parts: 256, perPart: 8, width: 1024 });
  });

  for (const given of DECIDED) {
    it(`gives the whole row's draw target bit for bit: ${given.name}`, async () => {
      const expected = samplerDistribution(
        given.row,
        given.spec ?? DECLARED,
        [],
        given.suppressed ?? [],
      );
      const actual = await fromSelection(given);
      assert(actual !== undefined, "the selection did not decide a row it holds");
      assertEquals([...actual.tokens], [...expected.tokens]);
      assertEquals(actual.probabilities, expected.probabilities);
    });
  }

  for (const given of UNDECIDED) {
    it(`does not decide, leaving the draw to the whole row: ${given.name}`, async () => {
      assertEquals(await fromSelection(given), undefined);
    });
  }
});
