/**
 * The decode run-ahead of the generation surface (`GenerationSequenceOptions.greedyAhead`), without
 * a GPU.
 *
 * The invariant: with the run-ahead the turn is **the per-step greedy decode's turn** — the same
 * token events at the same positions, the same stop, the same `used`, and the same frontier for the
 * next turn — because only the batching changes. Every case runs the same fake model (each row's
 * argmax is a fixed function of its input token) twice, once through the per-step greedy hook and
 * once through the fake run-ahead (chained deferred runs fed the previous step's selection), and
 * compares what a caller can observe.
 */
import { describe, expect, it } from "vite-plus/test";
import {
  createGenerationSequence,
  type GenerationGreedyRun,
  type GenerationRequest,
  type GenerationSequence,
  type GenerationSequenceOptions,
  type GenerationStop,
} from "../src/generation/sequence.ts";
import type { GenerationProgramSpec } from "../src/generation/program.ts";
import {
  drain,
  type FakeContext,
  fakeArgmax,
  type FakeOptions,
  type FakeSession,
  fakeSession,
  programOf,
  VOCAB,
} from "./helpers/generation-fake.ts";

/** Each row's argmax is the next id (`1 → 2 → 3 …`), so a turn's tokens are known in advance. */
const SUCCESSOR = Array.from({ length: VOCAB }, (_, id) => (id + 1) % VOCAB);

type Path = "per-step" | "ahead";

type Rig = { readonly fake: FakeSession; readonly sequence: GenerationSequence };

/**
 * A sequence on its own fake: `"per-step"` decodes through the greedy hook (one step per call),
 * `"ahead"` through the fake run-ahead of `depth` steps per batch.
 */
const open = async (
  path: Path,
  options: {
    readonly fake?: FakeOptions;
    readonly depth?: number;
    readonly program?: Partial<GenerationProgramSpec>;
    readonly sequence?: Partial<GenerationSequenceOptions<FakeContext>>;
  } = {},
): Promise<Rig> => {
  const fake = fakeSession({ successor: SUCCESSOR, ...options.fake });
  const greedy: GenerationGreedyRun<FakeContext> = async (inputs, generation) =>
    fakeArgmax(await fake.session.run(inputs, undefined, generation));
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, {
      derivedInputs: { ...fake.derivedInputs, positionsOnly: true },
      ...options.program,
    }),
    ...(path === "ahead" ? { greedyAhead: fake.greedyAhead(options.depth ?? 4) } : { greedy }),
    ...options.sequence,
  });
  return { fake, sequence };
};

/** The stop without the run-ahead counters (the one field the two paths legitimately differ in). */
const withoutCounters = (stop: GenerationStop): Omit<GenerationStop, "decodeAhead"> => {
  const { decodeAhead: _counters, ...rest } = stop;
  return rest;
};

/** One turn, as a caller sees it. */
const turn = async (rig: Rig, request: GenerationRequest) => {
  const { events, stop } = await drain(rig.sequence.generate(request));
  return { events, stop: withoutCounters(stop), used: rig.sequence.used };
};

/** The decode runs as a step sees them: the token fed, the position, the row it sits at. */
const decodeSteps = (fake: FakeSession) =>
  fake.calls
    .filter((call) => call.queryLength === 1 && call.idsShape[1] === 1)
    .map((call) => ({ ids: call.ids, positions: call.positions, pastBefore: call.pastBefore }));

/** The prefill runs (the frontier a turn prepends shows in their ids). */
const prefills = (fake: FakeSession) =>
  fake.calls
    .filter((call) => call.queryLength > 1)
    .map((call) => ({ ids: call.ids, pastBefore: call.pastBefore }));

describe("the decode run-ahead (greedyAhead)", () => {
  describe("the token ids", () => {
    for (const depth of [2, 3, 4]) {
      it(`are the per-step greedy decode's, batch after batch and across turns (depth ${depth})`, async () => {
        const plain = await open("per-step");
        const ahead = await open("ahead", { depth });
        try {
          for (const request of [
            { prompt: [1, 2, 3], maxNewTokens: 7 },
            { prompt: [5], maxNewTokens: 5 },
          ]) {
            expect(await turn(ahead, request)).toEqual(await turn(plain, request));
          }
          // Each step was fed the token the per-step loop fed, at the same position and row.
          expect(decodeSteps(ahead.fake)).toEqual(decodeSteps(plain.fake));
          expect(prefills(ahead.fake)).toEqual(prefills(plain.fake));
          expect(ahead.fake.pastLength()).toBe(plain.fake.pastLength());
          expect(ahead.fake.pendingCommit()).toBeUndefined();
        } finally {
          await plain.sequence.dispose();
          await ahead.sequence.dispose();
        }
      });
    }

    it("leave a turn whose counters name the batches and the steps", async () => {
      const ahead = await open("ahead", { depth: 4 });
      try {
        // 7 new tokens = the prefill's + 6 decode steps = a batch of 4 and the 2 the budget leaves.
        const { stop } = await drain(ahead.sequence.generate({ prompt: [1], maxNewTokens: 7 }));
        expect(ahead.fake.aheadBatches).toEqual([4, 2]);
        expect(stop.decodeAhead).toEqual({ depth: 4, batches: 2, steps: 6 });
      } finally {
        await ahead.sequence.dispose();
      }
    });
  });

  describe("a stop token in a batch", () => {
    // The prompt [1] prefills to the token 2; the first batch feeds 2 and selects 3, 4, 5, 6.
    for (const [at, stopToken] of [
      ["first", 3],
      ["middle", 4],
      ["last", 6],
    ] as const) {
      it(`at its ${at} step commits up to it and leaves the per-step frontier`, async () => {
        const plain = await open("per-step");
        const ahead = await open("ahead", { depth: 4 });
        try {
          const request = { prompt: [1], maxNewTokens: 16, stopTokens: [stopToken] };
          const stopped = await turn(ahead, request);
          expect(stopped).toEqual(await turn(plain, request));
          expect(stopped.stop).toMatchObject({ reason: "stop-token", token: stopToken });
          // The rows up to the stop are committed in one commit; the steps after it are not.
          expect(ahead.fake.commits).toEqual([stopToken - 2]);
          expect(ahead.fake.pastLength()).toBe(plain.fake.pastLength());
          // The next turn prepends the stop token (the frontier) exactly as after the per-step loop.
          const next = { prompt: [9], maxNewTokens: 3 };
          expect(await turn(ahead, next)).toEqual(await turn(plain, next));
          expect(prefills(ahead.fake)).toEqual(prefills(plain.fake));
        } finally {
          await plain.sequence.dispose();
          await ahead.sequence.dispose();
        }
      });
    }
  });

  describe("the token budget", () => {
    it("sizes the last batch to what is left and runs nothing for a one-token turn", async () => {
      for (const [maxNewTokens, batches] of [
        [7, [4, 2]],
        [5, [4]],
        [2, [1]],
        [1, []],
      ] as const) {
        const ahead = await open("ahead", { depth: 4 });
        try {
          const { stop } = await drain(ahead.sequence.generate({ prompt: [1], maxNewTokens }));
          expect(stop).toMatchObject({ reason: "max-tokens", tokens: maxNewTokens });
          expect(ahead.fake.aheadBatches).toEqual(batches);
        } finally {
          await ahead.sequence.dispose();
        }
      }
    });

    it("without maxNewTokens is the limit: the same tokens as an explicit one, no row past the capacity", async () => {
      const program = { capacity: 8 };
      const absent = await open("ahead", { depth: 4, program });
      const explicit = await open("ahead", { depth: 4, program });
      const plain = await open("per-step", { program });
      try {
        const unlimited = await turn(absent, { prompt: [1] });
        const limited = await turn(explicit, { prompt: [1], maxNewTokens: 8 });
        const perStep = await turn(plain, { prompt: [1] });

        expect(unlimited.stop).toEqual({ reason: "capacity", tokens: 8 });
        expect(limited.stop).toEqual({ reason: "max-tokens", tokens: 8 });
        expect(unlimited.events).toEqual(limited.events);
        expect(unlimited).toEqual(perStep);
        // 7 decode steps = a batch of 4 and the 3 the limit leaves.
        expect(absent.fake.aheadBatches).toEqual([4, 3]);
        expect(decodeSteps(absent.fake)).toEqual(decodeSteps(plain.fake));
        for (const call of absent.fake.calls) {
          expect(call.pastBefore + call.queryLength).toBeLessThanOrEqual(8);
        }
      } finally {
        await absent.sequence.dispose();
        await explicit.sequence.dispose();
        await plain.sequence.dispose();
      }
    });
  });

  describe("an interruption", () => {
    for (const how of ["break", "abort"] as const) {
      it(`by ${how} between the tokens of a batch keeps what the per-step loop kept`, async () => {
        const results = [];
        for (const path of ["per-step", "ahead"] as const) {
          const rig = await open(path, { depth: 4 });
          const control = new AbortController();
          try {
            const stream = rig.sequence.generate({
              prompt: [1],
              maxNewTokens: 8,
              signal: control.signal,
            });
            let delivered = 0;
            const consume = async (): Promise<void> => {
              for await (const event of stream) {
                if (event.kind !== "token") continue;
                // The third token is the second step of the first batch.
                if (++delivered < 3) continue;
                if (how === "break") break;
                control.abort(new Error("stop test"));
              }
            };
            if (how === "abort") await expect(consume()).rejects.toThrow("stop test");
            else await consume();
            const stopped = withoutCounters(await stream.done);
            const resumed = await turn(rig, { prompt: [12], maxNewTokens: 3 });
            results.push({ delivered, stopped, resumed, past: rig.fake.pastLength() });
          } finally {
            await rig.sequence.dispose();
          }
        }
        expect(results[1]).toEqual(results[0]);
      });
    }

    it("that arrives between batches stops before the next batch runs", async () => {
      const ahead = await open("ahead", { depth: 2 });
      const control = new AbortController();
      try {
        const stream = ahead.sequence.generate({
          prompt: [1],
          maxNewTokens: 8,
          signal: control.signal,
        });
        let delivered = 0;
        await expect(
          (async () => {
            for await (const event of stream) {
              // The prefill's token, then the two of the first batch: abort after the last one.
              if (event.kind === "token" && ++delivered === 3) control.abort(new Error("between"));
            }
          })(),
        ).rejects.toThrow("between");
        expect(ahead.fake.aheadBatches).toEqual([2]);
        expect(await stream.done).toMatchObject({ reason: "aborted", tokens: 3 });
      } finally {
        await ahead.sequence.dispose();
      }
    });
  });

  describe("a draw that throws (NaN logits)", () => {
    it("keeps the row its step consumed, as the per-step loop does", async () => {
      // Call 0 is the prefill; call 2 is the second decode step (inside the first batch).
      const results = [];
      for (const path of ["per-step", "ahead"] as const) {
        const rig = await open(path, { depth: 4, fake: { nanAt: 2 } });
        try {
          const failed = await drain(rig.sequence.generate({ prompt: [1], maxNewTokens: 8 })).then(
            () => undefined,
            (error: unknown) => (error instanceof Error ? error.message : String(error)),
          );
          const used = rig.sequence.used;
          const resumed = await turn(rig, { prompt: [7], maxNewTokens: 3 });
          results.push({ failed, used, resumed, past: rig.fake.pastLength() });
        } finally {
          await rig.sequence.dispose();
        }
      }
      expect(results[0].failed).toContain("NaN");
      expect(results[1]).toEqual(results[0]);
    });
  });

  describe("a failed batch", () => {
    it("commits none of its rows and keeps the frontier it was fed", async () => {
      // Call 2 fails: the second step of the first batch.
      const ahead = await open("ahead", { depth: 4, fake: { failAt: 2 } });
      try {
        await expect(
          drain(ahead.sequence.generate({ prompt: [1, 2], maxNewTokens: 8 })),
        ).rejects.toThrow();
        // The prefill's two rows only; the frontier (the prefill's token 3) is still pending.
        expect(ahead.fake.pastLength()).toBe(2);
        expect(ahead.fake.pendingCommit()).toBeUndefined();
        expect(ahead.sequence.used).toBe(3);
        await drain(ahead.sequence.generate({ prompt: [9], maxNewTokens: 1 }));
        expect(prefills(ahead.fake).at(-1)?.ids.slice(0, 2)).toEqual([3, 9]);
      } finally {
        await ahead.sequence.dispose();
      }
    });
  });

  describe("the path a turn takes", () => {
    it("stays per step for a sampler that is not plain greedy, and the counters say so", async () => {
      const ahead = await open("ahead", { depth: 4 });
      const plain = await open("per-step");
      try {
        const request = { prompt: [1], maxNewTokens: 6, sampler: { temperature: 1, seed: 3 } };
        const { stop } = await drain(ahead.sequence.generate(request));
        expect(stop.decodeAhead).toEqual({ depth: 4, batches: 0, steps: 0 });
        expect(ahead.fake.aheadBatches).toEqual([]);
        expect(withoutCounters(stop)).toEqual((await turn(plain, request)).stop);
      } finally {
        await ahead.sequence.dispose();
        await plain.sequence.dispose();
      }
    });

    it("stays per step on a sequence with run observation (onRun)", async () => {
      const ahead = await open("ahead", { depth: 4, sequence: { onRun: () => {} } });
      try {
        const { stop } = await drain(ahead.sequence.generate({ prompt: [1], maxNewTokens: 6 }));
        expect(stop.decodeAhead).toEqual({ depth: 4, batches: 0, steps: 0 });
        expect(ahead.fake.aheadBatches).toEqual([]);
      } finally {
        await ahead.sequence.dispose();
      }
    });

    it("leaves no counters on a sequence built without it", async () => {
      const plain = await open("per-step");
      try {
        const { stop } = await drain(plain.sequence.generate({ prompt: [1], maxNewTokens: 3 }));
        expect(Object.hasOwn(stop, "decodeAhead")).toBe(false);
      } finally {
        await plain.sequence.dispose();
      }
    });
  });

  describe("the setup", () => {
    const refused = async (message: string, options: Parameters<typeof open>[1]): Promise<void> => {
      const fake = fakeSession({ successor: SUCCESSOR, ...options?.fake });
      await expect(
        createGenerationSequence({
          session: fake.session,
          program: programOf(fake, {
            derivedInputs: { ...fake.derivedInputs, positionsOnly: true },
            ...options?.program,
          }),
          greedyAhead: fake.greedyAhead(options?.depth ?? 4),
          ...options?.sequence,
        }),
      ).rejects.toThrow(message);
      // The context of a refused setup is closed before the refusal reaches the caller.
      expect(fake.disposals()).toBe(1);
    };

    it("refuses a depth below 2", async () => {
      for (const depth of [1, 0, 2.5]) await refused("integer ≥ 2", { depth });
    });

    it("refuses a depth beyond the sliding slack or chunkLength", async () => {
      await refused("slack 3", { depth: 4, fake: { slidingSlack: 3 } });
      await refused("chunkLength 4", { depth: 5 });
    });

    it("refuses speculation next to it", async () => {
      await refused("cannot be combined", {
        sequence: {
          speculative: {
            policy: "always",
            open: () =>
              Promise.resolve({
                steps: 1,
                draft: () => Promise.resolve(Int32Array.of(1)),
                dispose: () => Promise.resolve(),
              }),
          },
        },
      });
    });

    it("refuses derived inputs that may read the token ids", async () => {
      const fake = fakeSession({ successor: SUCCESSOR });
      await expect(
        createGenerationSequence({
          session: fake.session,
          program: programOf(fake),
          greedyAhead: fake.greedyAhead(4),
        }),
      ).rejects.toThrow("positionsOnly");
      expect(fake.disposals()).toBe(1);
    });
  });
});
