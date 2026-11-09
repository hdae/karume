// Behaviour tests of the re-run face of the exact expert cache
// (`src/generation/expert-cache-session.ts` — ADR 0146 scheme A). Neither a GPU nor real assets are
// needed: the cache and the context are taken through narrow interfaces, so a scripted fake stands in.
//
// What this pins: a run that missed is dropped (`commit(0)`) before its experts are loaded and the
// same inputs run again; only a run that missed nothing is committed, touched and handed on; the
// outputs of a missed run never reach the caller (so the sampler never draws from them); the
// counters; the refusals of what the face owns; and that one protocol runs at a time.

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type {
  ExpertCacheStats,
  ExpertRouting,
  GenerationContext,
  RunInputs,
  RunOutputs,
  Session,
} from "@karume/runtime";
import {
  createExpertCacheSession,
  type ExpertCacheRuns,
  type ExpertCacheSession,
  type ExpertCacheTarget,
} from "../src/generation/expert-cache-session.ts";
import {
  createGenerationSequence,
  type GenerationContextFace,
  type GenerationSession,
} from "../src/generation/sequence.ts";
import {
  drain,
  type FakeContext,
  fakeSession,
  LOGITS,
  programOf,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

/** A context that records its commits in the shared log and holds one pending run like the real one. */
type LoggedContext = GenerationContextFace & {
  readonly label: string;
  defer(rows: number): void;
};

const loggedContext = (label: string, log: string[]): LoggedContext => {
  let pastLength = 0;
  let pending: { readonly pastLength: number; readonly queryLength: number } | undefined;
  return {
    label,
    get pastLength(): number {
      return pastLength;
    },
    get pendingCommit() {
      return pending;
    },
    slidingSlack: 8,
    defer: (rows) => {
      if (pending !== undefined) throw new Error(`${label}: a run awaits commit`);
      pending = { pastLength, queryLength: rows };
    },
    commit: (rows) => {
      if (pending === undefined) throw new Error(`${label}: nothing awaits commit`);
      log.push(`${label} commit(${rows})`);
      pastLength = pending.pastLength + rows;
      pending = undefined;
    },
    dispose: () => Promise.resolve(),
  };
};

const INPUTS: RunInputs = {
  input_ids: { dtype: "i32", shape: [1, 2], data: Int32Array.of(3, 4) },
};

/** The attempt a run's outputs came from (the scripted cache writes it into `logits`). */
const attemptOf = (outputs: RunOutputs): number => {
  const logits = outputs["logits"];
  assert(logits.dtype === "f32", "logits are f32");
  return logits.data[0];
};

/**
 * A cache whose runs miss as `misses` says (one entry per attempt, then 0): every run defers its
 * rows on the context it was given and returns outputs and a routing numbered by attempt.
 */
const scriptedCache = (
  misses: number[],
  log: string[],
  options: { readonly layers?: number; readonly failLoad?: boolean } = {},
) => {
  let attempts = 0;
  let missedRuns = 0;
  let loads = 0;
  const issued: unknown[] = [];
  const cache: ExpertCacheRuns<LoggedContext> = {
    slots: 4,
    layers: options.layers ?? 3,
    run: (_inputs, _bindings, generation) => {
      attempts += 1;
      const missing = misses.shift() ?? 0;
      if (missing > 0) missedRuns += 1;
      issued.push(generation);
      log.push(`${generation?.context.label ?? "stateless"} run#${attempts} missing ${missing}`);
      generation?.context.defer(generation.queryLength);
      const routing: ExpertRouting = {
        layers: 1,
        rows: generation?.queryLength ?? 2,
        picks: 1,
        ids: Int32Array.of(attempts, attempts),
      };
      return Promise.resolve({
        outputs: { logits: { dtype: "f32", shape: [1], data: Float32Array.of(attempts) } },
        routing,
        missing,
      });
    },
    load: (routing) => {
      log.push(`load [${[...routing.ids].join(",")}]`);
      if (options.failLoad === true) return Promise.reject(new Error("read failed"));
      loads += 1;
      return Promise.resolve({
        loaded: 1,
        evicted: 0,
        fromHost: 0,
        fromContainer: 1,
        bytes: 10,
        uploadedBytes: 10,
        ms: 1,
        readMs: 1,
        uploadMs: 0,
      });
    },
    touch: (routing, rows) => {
      log.push(`touch ${rows} [${[...routing.ids].join(",")}]`);
    },
    stats: (): ExpertCacheStats => ({
      runs: attempts,
      missedRuns,
      loads,
      loadedExperts: loads,
      evictedExperts: 0,
      hostExperts: 0,
      containerExperts: loads,
      bytesRead: 10 * loads,
      bytesUploaded: 10 * loads,
      loadMs: loads,
      readMs: loads,
      uploadMs: 0,
      hostHeldExperts: 3,
      hostHeldBytes: 30,
    }),
  };
  const target: ExpertCacheTarget<LoggedContext> = {
    createGenerationContext: () => Promise.reject(new Error("not used")),
    expertCache: cache,
  };
  return { target, issued };
};

describe("the expert cache's re-run face", () => {
  describe("a run that misses", () => {
    it("is dropped with commit(0) before the load, run again after it, and only the clean run is committed and touched", async () => {
      const log: string[] = [];
      const { target, issued } = scriptedCache([3, 1, 0], log);
      const face = createExpertCacheSession(target);
      const context = loggedContext("A", log);
      const outputs = await face.run(INPUTS, undefined, { context, queryLength: 2 });

      assertEquals(log, [
        "A run#1 missing 3",
        "A commit(0)",
        "load [1,1]",
        "A run#2 missing 1",
        "A commit(0)",
        "load [2,2]",
        "A run#3 missing 0",
        "A commit(2)",
        "touch 2 [3,3]",
      ]);
      // MUST: the outputs handed on are the clean attempt's, never a missed one's.
      assertEquals(attemptOf(outputs), 3);
      assertEquals(context.pastLength, 2);
      assertEquals(context.pendingCommit, undefined);
      // Every attempt runs deferred on the caller's context and rows, never chained.
      assertEquals(
        issued,
        [1, 2, 3].map(() => ({ context, queryLength: 2, commit: "deferred" })),
      );
    });

    it("counts the committed run and the re-runs, merged with the cache's own counters", async () => {
      const log: string[] = [];
      const { target } = scriptedCache([2, 0, 0, 1, 0], log);
      const face = createExpertCacheSession(target);
      const context = loggedContext("A", log);
      for (let run = 0; run < 3; run += 1) {
        await face.run(INPUTS, undefined, { context, queryLength: 1 });
      }
      assertEquals(face.report(), {
        runs: 5,
        missedRuns: 2,
        loads: 2,
        loadedExperts: 2,
        evictedExperts: 0,
        hostExperts: 0,
        containerExperts: 2,
        bytesRead: 20,
        bytesUploaded: 20,
        loadMs: 2,
        readMs: 2,
        uploadMs: 0,
        hostHeldExperts: 3,
        hostHeldBytes: 30,
        slots: 4,
        committedRuns: 3,
        retries: 2,
      });
    });
  });

  it("commits and touches a run that misses nothing on its first attempt, with no load", async () => {
    const log: string[] = [];
    const { target } = scriptedCache([0], log);
    const face = createExpertCacheSession(target);
    const outputs = await face.run(INPUTS, undefined, {
      context: loggedContext("A", log),
      queryLength: 1,
    });
    assertEquals(log, ["A run#1 missing 0", "A commit(1)", "touch 1 [1,1]"]);
    assertEquals(attemptOf(outputs), 1);
    assertEquals(face.report().retries, 0);
  });

  it("retries a run without a generation the same way and commits nothing", async () => {
    const log: string[] = [];
    const { target, issued } = scriptedCache([1, 0], log);
    const face = createExpertCacheSession(target);
    const outputs = await face.run(INPUTS, undefined);
    assertEquals(log, ["stateless run#1 missing 1", "load [1,1]", "stateless run#2 missing 0"]);
    assertEquals(attemptOf(outputs), 2);
    assertEquals(issued, [undefined, undefined]);
    assertEquals(face.report().committedRuns, 0);
    assertEquals(face.report().retries, 1);
  });

  it("passes a failed load to the caller with no pending commit left behind", async () => {
    const log: string[] = [];
    const { target } = scriptedCache([1], log, { failLoad: true });
    const face = createExpertCacheSession(target);
    const context = loggedContext("A", log);
    await assertRejects(
      () => face.run(INPUTS, undefined, { context, queryLength: 1 }),
      Error,
      "read failed",
    );
    assertEquals(log, ["A run#1 missing 1", "A commit(0)", "load [1,1]"]);
    assertEquals(context.pendingCommit, undefined);
    // The context takes the next run (nothing awaits commit).
    await face.run(INPUTS, undefined, { context, queryLength: 1 });
    assertEquals(context.pastLength, 1);
  });

  it("fails loudly, with no pending commit, when a run still misses after one attempt per routed layer and one more", async () => {
    const log: string[] = [];
    const { target } = scriptedCache([1, 1, 1, 1, 1], log, { layers: 2 });
    const face = createExpertCacheSession(target);
    const context = loggedContext("A", log);
    await assertRejects(
      () => face.run(INPUTS, undefined, { context, queryLength: 1 }),
      Error,
      "after 3 attempts",
    );
    assertEquals(log.filter((line) => line.includes("run#")).length, 3);
    assertEquals(log.at(-1), "A commit(0)");
    assertEquals(context.pendingCommit, undefined);
  });

  it("runs one protocol at a time: a second conversation's run waits for the first one's re-run and touch", async () => {
    const log: string[] = [];
    const { target } = scriptedCache([1, 0, 1, 0], log);
    const face = createExpertCacheSession(target);
    await Promise.all([
      face.run(INPUTS, undefined, { context: loggedContext("A", log), queryLength: 1 }),
      face.run(INPUTS, undefined, { context: loggedContext("B", log), queryLength: 1 }),
    ]);
    assertEquals(log, [
      "A run#1 missing 1",
      "A commit(0)",
      "load [1,1]",
      "A run#2 missing 0",
      "A commit(1)",
      "touch 1 [2,2]",
      "B run#3 missing 1",
      "B commit(0)",
      "load [3,3]",
      "B run#4 missing 0",
      "B commit(1)",
      "touch 1 [4,4]",
    ]);
  });

  describe("what the face owns", () => {
    it("refuses a caller's commit, chain or bidirectional rows before any run", async () => {
      const log: string[] = [];
      const { target } = scriptedCache([], log);
      const face = createExpertCacheSession(target);
      const context = loggedContext("A", log);
      await assertRejects(
        () => face.run(INPUTS, undefined, { context, queryLength: 1, commit: "deferred" }),
        Error,
        "commits a run's rows itself",
      );
      // `chain` is not on the face's type; a caller passing it anyway is refused all the same.
      const chained = { context, queryLength: 1, chain: true };
      await assertRejects(() => face.run(INPUTS, undefined, chained), Error, "cannot be chained");
      await assertRejects(
        () =>
          face.run(INPUTS, undefined, {
            context,
            queryLength: 2,
            bidirectionalRows: { start: 0, end: 2 },
          }),
        Error,
        "bidirectionalRows",
      );
      assertEquals(log, []);
      assertEquals(context.pendingCommit, undefined);
    });

    it("refuses a Session built without the cache", () => {
      assertThrows(
        () =>
          createExpertCacheSession({
            createGenerationContext: () => Promise.reject(new Error("not used")),
            expertCache: undefined,
          }),
        Error,
        "without SessionOptions.expertCache",
      );
    });
  });
});

describe("the face under the generation loop", () => {
  /** Next token = previous + 1, with a near-tie runner-up so a seeded draw depends on every logit. */
  const SUCCESSOR = Array.from({ length: VOCAB }, (_, token) => (token + 1) % VOCAB);
  const FAKE = { successor: SUCCESSOR, runnerUp: { id: 9, logit: 9.5 } } as const;
  const REQUEST = {
    prompt: [1, 2, 3],
    maxNewTokens: 4,
    sampler: { temperature: 1, seed: 11 },
  } as const;

  /** What a missed run would give: every logit on id 0 (another token than the clean run's). */
  const missedOutputs = (outputs: RunOutputs): RunOutputs => {
    const logits = outputs[LOGITS];
    assert(logits.dtype === "f32", "logits are f32");
    const data = new Float32Array(logits.data.length);
    for (let row = 0; row < data.length / VOCAB; row += 1) data[row * VOCAB] = 50;
    return { ...outputs, [LOGITS]: { ...logits, data } };
  };

  it("generates the tokens of the same model without the cache, and commits only clean runs", async () => {
    const reference = fakeSession(FAKE);
    const expected = await drain(
      (
        await createGenerationSequence({
          session: reference.session,
          program: programOf(reference),
        })
      ).generate(REQUEST),
    );

    const base = fakeSession(FAKE);
    // One entry per attempt: the prefill misses twice, decode steps 1 and 3 once.
    const misses = [2, 1, 0, 1, 0, 0, 1, 0];
    const cache: ExpertCacheRuns<FakeContext> = {
      slots: 16,
      layers: 4,
      run: async (inputs, bindings, generation) => {
        assert(generation !== undefined, "the generation loop always passes a generation");
        const outputs = await base.session.run(inputs, bindings, generation);
        const missing = misses.shift() ?? 0;
        return {
          outputs: missing > 0 ? missedOutputs(outputs) : outputs,
          routing: {
            layers: 1,
            rows: generation.queryLength,
            picks: 1,
            ids: new Int32Array(generation.queryLength),
          },
          missing,
        };
      },
      load: () =>
        Promise.resolve({
          loaded: 1,
          evicted: 0,
          fromHost: 1,
          fromContainer: 0,
          bytes: 0,
          uploadedBytes: 0,
          ms: 0,
          readMs: 0,
          uploadMs: 0,
        }),
      touch: () => {},
      stats: () => ({
        runs: 0,
        missedRuns: 0,
        loads: 0,
        loadedExperts: 0,
        evictedExperts: 0,
        hostExperts: 0,
        containerExperts: 0,
        bytesRead: 0,
        bytesUploaded: 0,
        loadMs: 0,
        readMs: 0,
        uploadMs: 0,
        hostHeldExperts: 0,
        hostHeldBytes: 0,
      }),
    };
    const face = createExpertCacheSession({
      createGenerationContext: (spec) => base.session.createGenerationContext(spec),
      expertCache: cache,
    });
    const sequence = await createGenerationSequence({ session: face, program: programOf(base) });
    const actual = await drain(sequence.generate(REQUEST));

    assertEquals(tokenIds(actual.events), tokenIds(expected.events));
    assertEquals(actual.stop, expected.stop);
    // Prefill: two drops then 3 rows; decode: a drop then 1, 1, a drop then 1.
    assertEquals(base.commits, [0, 0, 3, 0, 1, 1, 0, 1]);
    assertEquals(face.report().committedRuns, 4);
    assertEquals(face.report().retries, 4);
    assertEquals(misses, []);
  });
});

test("ExpertCacheTarget: the real Session satisfies it, and the face is a GenerationSession (detects spelling drift)", () => {
  // A type-check-only check (it does nothing at run time).
  const asTarget = (session: Session): ExpertCacheTarget<GenerationContext> => session;
  const asGenerationSession = (
    face: ExpertCacheSession<GenerationContext>,
  ): GenerationSession<GenerationContext> => face;
  assertEquals([typeof asTarget, typeof asGenerationSession], ["function", "function"]);
});
