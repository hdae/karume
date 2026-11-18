// Behaviour tests of the candidate selection in the generation loop
// (`GenerationSequenceOptions.candidateSelection` of `src/generation/sequence.ts`). Neither a GPU
// nor real assets are needed: the selection is a fake over the fake Session (`selectionOf`).
//
// What this pins: a turn whose sampler qualifies draws its decode tokens from the selection and
// delivers **the tokens the whole-row turn delivers**; a step whose entries do not decide the draw
// reads the whole row and still does; which of the two each step took is counted in the turn's
// stop; and the frontier bookkeeping around a run that failed and a draw that failed is the
// whole-row loop's.

import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { SamplerSpec } from "../src/generation/sampler.ts";
import {
  createGenerationSequence,
  type GenerationCandidateSelection,
} from "../src/generation/sequence.ts";
import {
  drain,
  type FakeOptions,
  type FakeSession,
  fakeSession,
  LOGITS,
  programOf,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

/** The first `count` ids of `row` by value descending, the smaller id first on a tie, NaN first. */
const leading = (row: Float32Array<ArrayBuffer>, count: number) => {
  const ids = Array.from({ length: row.length }, (_unused, id) => id).sort((left, right) => {
    const leftNaN = Number.isNaN(row[left]);
    const rightNaN = Number.isNaN(row[right]);
    if (leftNaN !== rightNaN) return leftNaN ? -1 : 1;
    if (!leftNaN && row[left] !== row[right]) return row[left] > row[right] ? -1 : 1;
    return left - right;
  });
  const indices = Int32Array.from(ids.slice(0, count));
  return { values: Float32Array.from(indices, (id) => row[id]), indices };
};

/**
 * A candidate selection over `fake`: the run is the fake Session's, the leading entries are ranked
 * on the host, and the whole row is the run's own logits. `undecided` makes `leading` hold nothing
 * for the decode steps it names (0-based).
 */
const selectionOf = (
  fake: FakeSession,
  options: { readonly maxCount?: number; readonly undecided?: (step: number) => boolean } = {},
) => {
  const seen = { runs: 0, wholeReads: 0, askedCounts: [] as number[] };
  const selection: GenerationCandidateSelection<FakeSession["context"]> = {
    maxCount: options.maxCount ?? VOCAB,
    run: async (inputs, generation, draw) => {
      const step = seen.runs;
      seen.runs += 1;
      const outputs = await fake.session.run(inputs, undefined, generation);
      const logits = outputs[LOGITS];
      assert(logits.dtype === "f32");
      const row = logits.data;
      return await draw({
        leading: (count) => {
          seen.askedCounts.push(count);
          return options.undecided?.(step) === true ? undefined : leading(row, count);
        },
        // deno-lint-ignore require-await
        whole: async () => {
          seen.wholeReads += 1;
          return row.slice();
        },
      });
    },
  };
  return { selection, seen };
};

/** A runner-up close enough to the target that a draw at temperature 1 takes either. */
const RUNNER_UP = { id: 11, logit: 9.5 };
const TOKENS = [5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 1, 2];
const FAKE: FakeOptions = { tokens: TOKENS, runnerUp: RUNNER_UP };
const SAMPLED: SamplerSpec = { temperature: 1, topK: 3, topP: 0.999, seed: 7 };
const REQUEST = { prompt: [1, 2], maxNewTokens: 10 };

/** One turn of `request` on a fresh fake, with or without a selection. */
const turn = async (
  request: Parameters<Awaited<ReturnType<typeof createGenerationSequence>>["generate"]>[0],
  options: {
    readonly selection?: Parameters<typeof selectionOf>[1];
    readonly fake?: FakeOptions;
    readonly onRun?: () => void;
  } = {},
) => {
  const fake = fakeSession(options.fake ?? FAKE);
  const made = options.selection === undefined ? undefined : selectionOf(fake, options.selection);
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
    ...(made === undefined ? {} : { candidateSelection: made.selection }),
    ...(options.onRun === undefined ? {} : { onRun: options.onRun }),
  });
  const { events, stop } = await drain(sequence.generate(request));
  return { fake, sequence, tokens: tokenIds(events), stop, seen: made?.seen };
};

describe("GenerationSequence: the candidate selection", () => {
  describe("on a turn whose sampler draws from the top-k", () => {
    it("delivers the tokens the whole-row turn delivers, for the same seed", async () => {
      for (const seed of [0, 7, 20261118]) {
        const request = { ...REQUEST, sampler: { ...SAMPLED, seed } };
        const whole = await turn(request);
        const selected = await turn(request, { selection: {} });
        assertEquals(selected.tokens, whole.tokens, `seed ${seed}`);
        // The draws are not all the target: the comparison covers the RNG, not only the argmax.
        assert(whole.tokens.includes(RUNNER_UP.id) || seed !== 7, "seed 7 draws the runner-up");
        assertEquals(selected.fake.calls, whole.fake.calls, `seed ${seed}`);
      }
    });

    it("runs every decode step through the selection, asking for top-k + 1 leading entries, and reads no whole row", async () => {
      const { seen, stop } = await turn({ ...REQUEST, sampler: SAMPLED }, { selection: {} });
      // Ten tokens: the first from the prefill's logits on the host, nine decode steps.
      assertEquals(seen, { runs: 9, wholeReads: 0, askedCounts: Array(9).fill(4) });
      assertEquals(stop, {
        reason: "max-tokens",
        tokens: 10,
        candidateSelection: { count: 4, steps: 9, fallbacks: 0 },
      });
    });
  });

  describe("on a step whose entries do not decide the draw", () => {
    it("reads the whole row for that step alone, and the turn's tokens are still the whole-row turn's", async () => {
      const request = { ...REQUEST, sampler: SAMPLED };
      const whole = await turn(request);
      const selected = await turn(request, {
        selection: { undecided: (step) => step === 2 || step === 5 },
      });
      assertEquals(selected.tokens, whole.tokens);
      assertEquals(selected.seen?.wholeReads, 2);
      assertEquals(selected.stop, {
        reason: "max-tokens",
        tokens: 10,
        candidateSelection: { count: 4, steps: 7, fallbacks: 2 },
      });
    });

    it("counts a step the sampler itself gives up as a fallback: a tie under a temperature that reaches past the entries", async () => {
      // Every id but the target and the runner-up is at logit 0: at temperature 2 the third
      // candidate ties with everything below it, which the four leading entries cannot settle.
      const request = { ...REQUEST, sampler: { ...SAMPLED, temperature: 2 } };
      const whole = await turn(request);
      const selected = await turn(request, { selection: {} });
      assertEquals(selected.tokens, whole.tokens);
      assertEquals(selected.stop, {
        reason: "max-tokens",
        tokens: 10,
        candidateSelection: { count: 4, steps: 0, fallbacks: 9 },
      });
    });
  });

  describe("on a turn the selection does not serve", () => {
    it("decodes as without it and says so with a count of 0", async () => {
      for (const [name, sampler, options] of [
        ["no top-k", { temperature: 1, seed: 7 }, {}],
        ["a repetition penalty", { ...SAMPLED, repetitionPenalty: 1.3 }, {}],
        ["a logit bias", { ...SAMPLED, logitBias: [[5, -1]] }, {}],
        ["more entries than the selection offers", SAMPLED, { maxCount: 3 }],
      ] as const satisfies readonly (readonly [string, SamplerSpec, { maxCount?: number }])[]) {
        const request = { ...REQUEST, sampler };
        const whole = await turn(request);
        const selected = await turn(request, { selection: options });
        assertEquals(selected.tokens, whole.tokens, name);
        assertEquals(selected.seen?.runs, 0, name);
        assertEquals(
          selected.stop,
          { ...whole.stop, candidateSelection: { count: 0, steps: 0, fallbacks: 0 } },
          name,
        );
      }
    });

    it("leaves a turn observed run by run on the whole-row path (the hook reads that run's own outputs)", async () => {
      let observed = 0;
      const { seen, stop } = await turn(
        { ...REQUEST, sampler: SAMPLED },
        { selection: {}, onRun: () => (observed += 1) },
      );
      assertEquals(seen?.runs, 0);
      assertEquals(observed, 10);
      assertEquals(stop.candidateSelection, { count: 0, steps: 0, fallbacks: 0 });
    });
  });

  describe("on a turn at temperature 0", () => {
    it("reports nothing: no step draws, and the stop is the one a sequence without the selection gives", async () => {
      for (const sampler of [
        undefined,
        { topK: 3 },
        { temperature: 0, topK: 3, topP: 0.9 },
      ] as const) {
        const request = { ...REQUEST, ...(sampler === undefined ? {} : { sampler }) };
        const whole = await turn(request);
        const selected = await turn(request, { selection: {} });
        assertEquals(selected.tokens, whole.tokens);
        assertEquals(selected.seen?.runs, 0);
        assertEquals(Object.hasOwn(selected.stop, "candidateSelection"), false);
        assertEquals(selected.stop, whole.stop);
        assertEquals(selected.stop, { reason: "max-tokens", tokens: 10 });
      }
    });
  });

  it("is not reported by a sequence built without it", async () => {
    const { stop } = await turn({ ...REQUEST, sampler: SAMPLED });
    assertEquals(stop, { reason: "max-tokens", tokens: 10 });
  });

  describe("when a decode step fails", () => {
    // The run numbers are the fake's: 0 is the prefill, 1 the first decode step.
    it("keeps the frontier of a run that failed, so the next turn feeds it once", async () => {
      const request = { prompt: [1, 2], maxNewTokens: 4, sampler: SAMPLED };
      const options: FakeOptions = { ...FAKE, failAt: 2 };
      const a = fakeSession(options);
      const b = fakeSession(options);
      const whole = await createGenerationSequence({ session: a.session, program: programOf(a) });
      const selected = await createGenerationSequence({
        session: b.session,
        program: programOf(b),
        candidateSelection: selectionOf(b).selection,
      });
      await assertRejects(() => drain(whole.generate(request)), Error, "run failed");
      await assertRejects(() => drain(selected.generate(request)), Error, "run failed");
      assertEquals(selected.used, whole.used);
      // The next turn of both prepends the same frontier and runs the same rows.
      await drain(selected.generate({ ...request, prompt: [3] }));
      await drain(whole.generate({ ...request, prompt: [3] }));
      assertEquals(b.calls, a.calls);
    });

    it("drops the frontier of a run that went through when its draw fails, as the whole-row loop does: a NaN row", async () => {
      const request = { prompt: [1, 2], maxNewTokens: 4, sampler: SAMPLED };
      const options: FakeOptions = { ...FAKE, nanAt: 2 };
      const a = fakeSession(options);
      const b = fakeSession(options);
      const whole = await createGenerationSequence({ session: a.session, program: programOf(a) });
      const made = selectionOf(b);
      const selected = await createGenerationSequence({
        session: b.session,
        program: programOf(b),
        candidateSelection: made.selection,
      });
      const expected = await assertRejects(() => drain(whole.generate(request)), Error, "is NaN");
      const actual = await assertRejects(() => drain(selected.generate(request)), Error);
      // The entries held the NaN and did not decide; the whole row threw what it always throws.
      assertEquals(actual.message, expected.message);
      assertEquals(made.seen.wholeReads, 1);
      assertEquals(selected.used, whole.used);
      await drain(selected.generate({ ...request, prompt: [3] }));
      await drain(whole.generate({ ...request, prompt: [3] }));
      assertEquals(b.calls, a.calls);
    });

    it("fails loudly on a whole row of the wrong size, after the frontier was dropped", async () => {
      const fake = fakeSession(FAKE);
      const made = selectionOf(fake, { undecided: () => true });
      const selection: GenerationCandidateSelection<FakeSession["context"]> = {
        maxCount: made.selection.maxCount,
        run: (inputs, generation, draw) =>
          made.selection.run(inputs, generation, (row) =>
            draw({
              leading: (count) => row.leading(count),
              whole: async () => (await row.whole()).subarray(1),
            }),
          ),
      };
      const sequence = await createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        candidateSelection: selection,
      });
      await assertRejects(
        () => drain(sequence.generate({ ...REQUEST, sampler: SAMPLED })),
        Error,
        `the whole logits row has ${VOCAB - 1} values, not ${VOCAB}`,
      );
      // The prompt's two rows and the failed step's one are in the conversation; no frontier is.
      assertEquals(sequence.used, 3);
    });
  });
});
