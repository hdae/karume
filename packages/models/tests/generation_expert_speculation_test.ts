// Behaviour tests of self-speculation over the exact expert cache (ADR 0146 decision 8, scheme B —
// `src/generation/expert-speculation.ts`, driven by `sequence.ts` through the cache's face). Neither
// a GPU nor real assets are needed: the target model is the generation fake (each row's argmax is
// the successor of its input token), and the cache around it is scripted — which verify attempts
// miss from which row, and which drafts point at another token.
//
// What this pins: the tokens and the stop are plain decode's (at temperature 0 and with seeded draws
// at temperature 1), whatever the drafts and the misses do; a verify row at or after its first
// missing row is never drawn from (the fake makes those rows point at another token, so drawing one
// changes the tokens); the drafts are chained deferred runs dropped by one commit(0) and loaded as
// one routing; a verify whose first row misses is dropped, its routing loaded and run again; a cycle
// that reaches the first missing row commits the rows before it, touches them and loads the verify's
// routing; stop tokens in a draft and at a verify row; the last token of the budget as one verify
// row; the sampler draws once per confirmed token; a consumer's break commits the delivered rows only.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type { ExpertCacheStats, ExpertRouting, RunInputs, RunOutputs } from "@karume/runtime";
import {
  createExpertCacheSession,
  type ExpertCacheRuns,
} from "../src/generation/expert-cache-session.ts";
import { createSampler } from "../src/generation/sampler.ts";
import {
  createGenerationSequence,
  type GenerationRequest,
  type GenerationRunPhase,
} from "../src/generation/sequence.ts";
import type { GenerationProgramSpec } from "../src/generation/program.ts";
import {
  drain,
  type FakeContext,
  type FakeOptions,
  fakeSession,
  IDS,
  LAST_ROW,
  LOGITS,
  programOf,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

/** Each row's argmax is the next id (`3 → 4 → 5 …`), so a turn's tokens are known in advance. */
const SUCCESSOR = Array.from({ length: VOCAB }, (_, token) => (token + 1) % VOCAB);
/** A near-tie runner-up, so that a seeded draw at temperature 1 depends on its uniform value. */
const SAMPLED: FakeOptions = { successor: SUCCESSOR, runnerUp: { id: 9, logit: 9.5 } };
const GREEDY: FakeOptions = { successor: SUCCESSOR };
const PROMPT = [1, 2, 3];
/** The token a corrupted row points at instead of its own argmax. */
const wrongOf = (token: number): number => (token + 5) % VOCAB;

type Scenario = {
  /** Per verify position, each attempt's first missing row (absent = every row resident). */
  readonly verify?: Readonly<Record<number, readonly number[]>>;
  /** Draft runs (0-based, in issue order) whose logits point at another token than the model's. */
  readonly wrongDrafts?: readonly number[];
  readonly depth?: number;
  readonly fake?: FakeOptions;
  readonly program?: Partial<GenerationProgramSpec>;
  readonly onRun?: (phase: GenerationRunPhase) => void;
};

/** The one-hot row of `token` (what a corrupted row holds). */
const oneHot = (data: Float32Array<ArrayBuffer>, row: number, token: number): void => {
  data.fill(0, row * VOCAB, (row + 1) * VOCAB);
  data[row * VOCAB + token] = 20;
};

const argmaxOf = (data: Float32Array<ArrayBuffer>, row: number): number => {
  let best = 0;
  for (let token = 1; token < VOCAB; token += 1) {
    if (data[row * VOCAB + token] > data[row * VOCAB + best]) best = token;
  }
  return best;
};

const ZERO_STATS: ExpertCacheStats = {
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
};

/**
 * A scripted cache over the fake model: every run goes to the fake (which keeps the logical length
 * and the pending rows like the real context), the routing of a run is its input ids in two layers,
 * and the rows a scenario makes miss — a verify's rows from its first missing row, a wrong draft's
 * row — point at another token. `log` records the cache's calls in order.
 */
const scriptedCache = (scenario: Scenario) => {
  const base = fakeSession(scenario.fake ?? GREEDY);
  const log: string[] = [];
  const attempts = new Map<number, number>();
  let draftRuns = 0;
  const cache: ExpertCacheRuns<FakeContext> = {
    slots: 64,
    layers: 2,
    run: async (inputs, bindings, generation) => {
      assert(generation !== undefined, "every run under the cache is a generation run");
      const outputs = await base.session.run(inputs, bindings, generation);
      const call = base.calls.at(-1);
      assert(call !== undefined);
      const rows = generation.queryLength;
      const logits = outputs[LOGITS];
      assert(logits.dtype === "f32");
      const data = logits.data.slice();
      let first = rows;
      if (generation.chain === true) {
        // A draft runs on the cache as it is (whether it missed changes nothing the cycle reads).
        log.push(`draft@${call.pastBefore}`);
        if (scenario.wrongDrafts?.includes(draftRuns) === true)
          oneHot(data, 0, wrongOf(argmaxOf(data, 0)));
        draftRuns += 1;
      } else {
        const attempt = attempts.get(call.pastBefore) ?? 0;
        attempts.set(call.pastBefore, attempt + 1);
        first = scenario.verify?.[call.pastBefore]?.[attempt] ?? rows;
        log.push(`run@${call.pastBefore} rows ${rows} first ${first}`);
        call.lastRows.forEach((inputRow, row) => {
          if (inputRow >= first) oneHot(data, row, wrongOf(argmaxOf(data, row)));
        });
      }
      const ids = new Int32Array(2 * rows);
      for (let at = 0; at < ids.length; at += 1) ids[at] = call.ids[at % rows];
      const routing: ExpertRouting = { layers: 2, rows, picks: 1, ids };
      return {
        outputs: { ...outputs, [LOGITS]: { ...logits, data } },
        routing,
        missing: first < rows ? 1 : 0,
        firstMissingRow: first,
      };
    },
    load: (routing) => {
      log.push(`load ${routing.rows} [${[...routing.ids.subarray(0, routing.rows)].join(",")}]`);
      return Promise.resolve({
        loaded: 1,
        evicted: 0,
        fromHost: 1,
        fromContainer: 0,
        bytes: 0,
        uploadedBytes: 0,
        ms: 0,
        readMs: 0,
        uploadMs: 0,
      });
    },
    touch: (_routing, rows) => {
      log.push(`touch ${rows}`);
    },
    stats: () => ZERO_STATS,
  };
  return { base, log, cache };
};

/** A sequence over the scripted cache with self-speculation of `depth` (default 3). */
const cachedSequence = async (scenario: Scenario) => {
  const scripted = scriptedCache(scenario);
  const face = createExpertCacheSession(
    {
      createGenerationContext: (spec) => scripted.base.session.createGenerationContext(spec),
      expertCache: scripted.cache,
    },
    { depth: scenario.depth ?? 3 },
  );
  assert(face.speculation !== undefined, "a face with a depth speculates");
  const sequence = await createGenerationSequence({
    session: face,
    program: programOf(scripted.base, scenario.program),
    selfSpeculation: face.speculation,
    ...(scenario.onRun === undefined ? {} : { onRun: scenario.onRun }),
  });
  return { ...scripted, face, sequence };
};

/** The same turn by plain decode on the fake alone (one run per token). */
const plainTurn = async (scenario: Scenario, request: GenerationRequest) => {
  const base = fakeSession(scenario.fake ?? GREEDY);
  const sequence = await createGenerationSequence({
    session: base.session,
    program: programOf(base, scenario.program),
  });
  const turn = await drain(sequence.generate(request));
  return { ...turn, used: sequence.used };
};

/** Runs one turn both ways and checks that the cache's is plain decode's. */
const assertPlainTurn = async (scenario: Scenario, request: GenerationRequest) => {
  const expected = await plainTurn(scenario, request);
  const cached = await cachedSequence(scenario);
  const actual = await drain(cached.sequence.generate(request));
  assertEquals(tokenIds(actual.events), tokenIds(expected.events), "tokens");
  assertEquals(actual.events, expected.events, "events and positions");
  assertEquals(actual.stop, expected.stop, "stop");
  assertEquals(cached.sequence.used, expected.used, "used");
  const report = cached.face.report();
  assert(report.speculation !== undefined, "a face with a depth reports its cycles");
  return { ...cached, events: actual.events, speculation: report.speculation, report };
};

const greedy = (maxNewTokens: number): GenerationRequest => ({
  prompt: PROMPT,
  maxNewTokens,
  sampler: { temperature: 0 },
});

describe("self-speculation over the expert cache: one cycle", () => {
  it("drafts on the cache as it is, drops the drafts with one commit(0), loads them as one routing, and confirms every draft and one more token", async () => {
    const phases: GenerationRunPhase[] = [];
    const run = await assertPlainTurn({ onRun: (phase) => phases.push(phase) }, greedy(9));
    assertEquals(tokenIds(run.events), [4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assertEquals(run.log, [
      // The prefill: one scheme-A run, committed and touched.
      "run@0 rows 3 first 3",
      "touch 3",
      // Cycle 1 from frontier 4 at position 3: drafts 5, 6, 7 at 3, 4, 5.
      "draft@3",
      "draft@4",
      "draft@5",
      "load 3 [4,5,6]",
      "run@3 rows 4 first 4",
      "touch 4",
      // Cycle 2 from frontier 8 at position 7.
      "draft@7",
      "draft@8",
      "draft@9",
      "load 3 [8,9,10]",
      "run@7 rows 4 first 4",
      "touch 4",
    ]);
    // The prefill's rows, the draft chain dropped, the verify's rows as they were delivered.
    assertEquals(run.base.commits, [3, 0, 4, 0, 4]);
    // The drafts are chained deferred single rows; the verify one deferred run of depth + 1 rows.
    const decode = run.base.calls.slice(1);
    assertEquals(
      decode.map((call) => [call.ids.length, call.lastRows, call.commit, call.queryLength]),
      [
        [1, [0], "deferred", 1],
        [1, [0], "deferred", 1],
        [1, [0], "deferred", 1],
        [4, [0, 1, 2, 3], "deferred", 4],
        [1, [0], "deferred", 1],
        [1, [0], "deferred", 1],
        [1, [0], "deferred", 1],
        [4, [0, 1, 2, 3], "deferred", 4],
      ],
    );
    assertEquals(decode[3].positions, [3, 4, 5, 6]);
    assertEquals(run.speculation, {
      depth: 3,
      cycles: 2,
      draftRuns: 6,
      acceptedDrafts: 6,
      verifyRetries: 0,
      partialCommits: 0,
    });
    // Scheme A ran the prefill alone.
    assertEquals([run.report.committedRuns, run.report.retries], [1, 0]);
    assertEquals(
      phases.map((phase) =>
        phase.kind === "verify"
          ? [phase.kind, phase.cycle, phase.rows, phase.accepted, phase.delivered]
          : [phase.kind],
      ),
      [["prefill"], ["verify", 1, 4, 3, 4], ["verify", 2, 4, 3, 4]],
    );
  });

  it("drops a verify whose first row missed with commit(0), loads its routing and runs it again", async () => {
    const run = await assertPlainTurn({ verify: { 3: [0, 4] } }, greedy(5));
    assertEquals(run.log.slice(2), [
      "draft@3",
      "draft@4",
      "draft@5",
      "load 3 [4,5,6]",
      "run@3 rows 4 first 0",
      "load 4 [4,5,6,7]",
      "run@3 rows 4 first 4",
      "touch 4",
    ]);
    // The retry's drop comes before its load (a failed load leaves nothing pending).
    assertEquals(run.base.commits, [3, 0, 0, 4]);
    assertEquals([run.speculation.verifyRetries, run.speculation.cycles], [1, 1]);
  });

  it("keeps the rows before the verify's first missing row, touches them, and loads the verify's routing for the next cycle", async () => {
    const run = await assertPlainTurn({ verify: { 3: [2] } }, greedy(9));
    assertEquals(run.log.slice(2, 9), [
      "draft@3",
      "draft@4",
      "draft@5",
      "load 3 [4,5,6]",
      "run@3 rows 4 first 2",
      // Rows 0 and 1 confirmed 5 and 6 (both the drafts); row 2 may have missed and is not drawn.
      "touch 2",
      "load 4 [4,5,6,7]",
    ]);
    // The next cycle starts from frontier 6 at position 5.
    assertEquals(run.log[9], "draft@5");
    assertEquals(run.base.commits.slice(0, 3), [3, 0, 2]);
    assertEquals(run.speculation, {
      depth: 3,
      cycles: 3,
      draftRuns: 7,
      acceptedDrafts: 6,
      verifyRetries: 0,
      partialCommits: 1,
    });
  });

  it("ends a cycle at the draw of the row whose draft it rejected", async () => {
    // The second draft points at 11 instead of 6, so row 1 draws 6 and the cycle confirms 5, 6.
    const run = await assertPlainTurn({ wrongDrafts: [1] }, greedy(5));
    assertEquals(run.log.slice(2, 8), [
      "draft@3",
      "draft@4",
      "draft@5",
      "load 3 [4,5,11]",
      "run@3 rows 4 first 4",
      "touch 2",
    ]);
    assertEquals(run.base.commits.slice(0, 3), [3, 0, 2]);
    assertEquals(run.speculation, {
      depth: 3,
      cycles: 2,
      draftRuns: 4,
      acceptedDrafts: 2,
      verifyRetries: 0,
      partialCommits: 0,
    });
  });

  it("stops drafting at a stop token, and ends the turn when the verify confirms it", async () => {
    const run = await assertPlainTurn({ program: { stopTokens: [6] } }, greedy(9));
    assertEquals(run.events.filter((event) => event.kind === "token").length, 2);
    // Two drafts (5, then the stop 6), a verify of three rows in the fixed shape of four.
    assertEquals(run.log.slice(2), [
      "draft@3",
      "draft@4",
      "load 2 [4,5]",
      "run@3 rows 3 first 3",
      "touch 2",
    ]);
    assertEquals(run.base.calls.at(-1)?.lastRows, [0, 1, 2, 2]);
    assertEquals(run.base.commits, [3, 0, 2]);
  });

  it("ends the turn at a stop token an exact verify row draws", async () => {
    // The first draft points at 10, so row 0 draws the stop 5 against it.
    const run = await assertPlainTurn(
      { program: { stopTokens: [5] }, wrongDrafts: [0] },
      greedy(9),
    );
    assertEquals(tokenIds(run.events), [4]);
    assertEquals(run.log.at(-1), "touch 1");
    assertEquals(run.base.commits, [3, 0, 1]);
    assertEquals(run.speculation.acceptedDrafts, 0);
  });

  it("confirms the last token of the budget with one verify row in the decode shape", async () => {
    const run = await assertPlainTurn({}, greedy(2));
    assertEquals(run.log.slice(2), ["run@3 rows 1 first 1", "touch 1"]);
    const last = run.base.calls.at(-1);
    assertEquals([last?.idsShape, last?.lastRows, last?.commit], [[1, 1], [0], "deferred"]);
    assertEquals(run.speculation.draftRuns, 0);
    assertEquals(run.base.commits, [3, 1]);
  });
});

describe("self-speculation over the expert cache: the draws", () => {
  it("gives plain decode's seeded tokens at temperature 1 through retries, partial commits and rejections", async () => {
    const run = await assertPlainTurn(
      { fake: SAMPLED, verify: { 3: [0, 2] }, wrongDrafts: [3] },
      { prompt: PROMPT, maxNewTokens: 16, sampler: { temperature: 1, seed: 11 } },
    );
    const { speculation } = run;
    // MUST (precondition): the turn went through each path.
    assert(speculation.verifyRetries > 0 && speculation.partialCommits > 0);
    assert(speculation.acceptedDrafts > 0 && speculation.acceptedDrafts < speculation.draftRuns);
    // MUST (precondition): the seeded draws took the runner-up somewhere (the uniforms decide).
    assert(tokenIds(run.events).includes(9), `${tokenIds(run.events).join(",")}`);
  });

  it("draws once per confirmed token and peeks once per draft, never consuming the RNG to peek", async () => {
    const scripted = scriptedCache({ fake: SAMPLED, verify: { 0: [3] } });
    const face = createExpertCacheSession(
      {
        createGenerationContext: (spec) => scripted.base.session.createGenerationContext(spec),
        expertCache: scripted.cache,
      },
      { depth: 3 },
    );
    const context = await face.createGenerationContext({ bindings: { C: 64 }, chunkLength: 4 });
    const counted = { next: 0, peek: 0 };
    const inner = createSampler({ temperature: 1, seed: 5 });
    const prepare = (ids: readonly number[]) => {
      const rows = ids.length === 1 ? 1 : 4;
      const tokens = new Int32Array(rows);
      tokens.set(ids);
      const selected = ids.length === 1 ? [0] : [0, 1, 2, 3];
      const inputs: RunInputs = {
        [IDS]: { dtype: "i32", shape: [1, rows], data: tokens },
        [LAST_ROW]: { dtype: "i32", shape: [selected.length], data: Int32Array.from(selected) },
      };
      return Promise.resolve({
        inputs,
        logits: (outputs: RunOutputs) => (row: number) => {
          const logits = outputs[LOGITS];
          assert(logits.dtype === "f32");
          return logits.data.subarray(row * VOCAB, (row + 1) * VOCAB);
        },
      });
    };
    assert(face.speculation !== undefined);
    const cycle = await face.speculation.cycle({
      context,
      token: 4,
      drafts: 3,
      history: [4],
      sampler: {
        next: (logits, history) => {
          counted.next += 1;
          return inner.next(logits, history);
        },
        peek: (logits, history, ahead) => {
          counted.peek += 1;
          return inner.peek(logits, history, ahead);
        },
      },
      isStop: () => false,
      prepare,
    });
    // Row 3 missed: rows 0..2 are drawn at most, once each, and nothing else.
    assert(cycle.confirmed.length >= 1 && cycle.confirmed.length <= 3);
    assertEquals(counted, { next: cycle.confirmed.length, peek: 3 });
    // The same seed's plain draws give the confirmed tokens (the peeks consumed nothing).
    const plain = createSampler({ temperature: 1, seed: 5 });
    const logitsOf = (token: number): Float32Array<ArrayBuffer> => {
      const data = new Float32Array(VOCAB);
      data[SUCCESSOR[token]] = 10;
      data[9] = 9.5;
      return data;
    };
    let fed = 4;
    const expected: number[] = [];
    for (let at = 0; at < cycle.confirmed.length; at += 1) {
      fed = plain.next(logitsOf(fed), []);
      expected.push(fed);
    }
    assertEquals(cycle.confirmed, expected);
    // The verify stays pending for the loop to commit (rows of the confirmed tokens at least).
    assert((context.pendingCommit?.queryLength ?? 0) >= cycle.confirmed.length);
  });
});

describe("self-speculation over the expert cache: the conversation", () => {
  it("commits only the rows of the tokens a consumer received when it breaks mid-cycle, as plain decode does", async () => {
    const cached = await cachedSequence({});
    const received: number[] = [];
    for await (const event of cached.sequence.generate(greedy(9))) {
      if (event.kind !== "token") continue;
      received.push(event.id);
      // 4 came from the prefill, 5 is the first of cycle 1's four confirmed tokens.
      if (event.id === 5) break;
    }
    assertEquals(received, [4, 5]);
    assertEquals(cached.base.commits, [3, 0, 1]);
    assertEquals([cached.base.pastLength(), cached.sequence.used], [4, 5]);
    // The next turn starts from the frontier 5 at position 4, as after plain decode.
    const next = await drain(cached.sequence.generate({ ...greedy(2), prompt: [7] }));
    assertEquals(tokenIds(next.events), [8, 9]);
    // The next turn's prefill (its last cycle is one verify row after it).
    const prefill = cached.base.calls.at(-2);
    assertEquals([prefill?.pastBefore, prefill?.ids.slice(0, 2)], [4, [5, 7]]);
  });
});

describe("self-speculation over the expert cache: the setup", () => {
  it("refuses a depth whose verify does not fit the chunk rows or the sliding slack, and a bad depth", async () => {
    await assertRejects(() => cachedSequence({ depth: 4 }), Error, "more than chunkLength 4");
    await assertRejects(
      () => cachedSequence({ depth: 3, fake: { ...GREEDY, slidingSlack: 3 } }),
      Error,
      "more than the sliding ring's slack 3",
    );
    const { base, cache } = scriptedCache({});
    for (const depth of [0, 1.5, Number.NaN]) {
      assertThrows(
        () =>
          createExpertCacheSession(
            {
              createGenerationContext: (spec) => base.session.createGenerationContext(spec),
              expertCache: cache,
            },
            { depth },
          ),
        Error,
        "is not a positive integer",
      );
    }
  });

  it("refuses self-speculation with a drafter, before the drafter opens", async () => {
    const { base, cache } = scriptedCache({});
    const face = createExpertCacheSession(
      {
        createGenerationContext: (spec) => base.session.createGenerationContext(spec),
        expertCache: cache,
      },
      { depth: 3 },
    );
    assert(face.speculation !== undefined);
    let opened = false;
    await assertRejects(
      () =>
        createGenerationSequence({
          session: face,
          program: programOf(base),
          selfSpeculation: face.speculation,
          speculative: {
            open: () => {
              opened = true;
              return Promise.reject(new Error("not used"));
            },
          },
        }),
      Error,
      "selfSpeculation and speculative cannot be combined",
    );
    assertEquals([opened, base.disposals()], [false, 1]);
  });

  it("reports no cycles on a face without a depth", () => {
    const { base, cache } = scriptedCache({});
    const face = createExpertCacheSession({
      createGenerationContext: (spec) => base.session.createGenerationContext(spec),
      expertCache: cache,
    });
    assertEquals(
      [face.speculation, Object.hasOwn(face.report(), "speculation")],
      [undefined, false],
    );
  });
});
