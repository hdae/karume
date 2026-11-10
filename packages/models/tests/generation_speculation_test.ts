// Behaviour tests of speculative decoding (ADR 0096 step 3). Neither a GPU nor real assets are
// needed.
//
// What this pins is the contract that "speculation changes only the speed and **does not change the
// output**":
//
// - The token sequence, positions, stop reason and the `tokens` count match **a reference run of
//   the same request without speculation** exactly (with 0 accepted, all accepted, or partly
//   accepted). A regression that breaks this raises no exception and shows up only as "the text
//   sometimes changes". **Independent of the temperature** — acceptance calls `sampler.next` once
//   per row (once per committed token) with the same logits, the same history and in the same order
//   as non-speculative decode, so the RNG consumption and the sequence both match (T3′).
// - Only "the tokens the consumer received + one frontier" enter the conversation (the row count
//   of `commit(rows)`). Rejected rows, committed candidates after a stop token, and tokens not yet
//   delivered after a `break` do not enter the KV.
// - No path exits with a deferred run still pending (stop, `break`, exception or interruption
//   alike). A context with a pending run accepts only dispose, so the next turn would surely fail.
//
// The comparison with the reference run can use "the same fake and the same successor function"
// because the fake's logits are set by {@link FakeOptions.successor} (the row's input token → the
// row's argmax). If the implementation mixed up the verify rows, a sequence different from the
// reference run would come out.
//
// T1–T15 are the contract **without the gate** (`policy: "always"`) — when the self-accounting gate
// (step 4-B ④) cuts in, the accounting of 1 cycle = draft + verify breaks down and it depends on
// the wall clock too. The gate is checked by T16 with a fake clock (the gate's own decisions are in
// `generation_gate_test.ts`).
//
// T17 covers **the wall time and the phase** (`wallMs` / `delivered` / `gate`) that the observation
// hook carries. The checks that look only at the run kind and number (T7 / T9) compare with the
// wall fields dropped by `runShape` — a fake clock cannot be injected into a non-speculative
// sequence, so those checks cannot see the actual wall values.
//
// T18 is the speculation side of forking (`fork`) — the drafter is reopened on the duplicated
// context, and the gate copies its observation state (ADR 0083 addendum 2026-10-11).

import { describe, it, test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import type { GenerationProgramSpec } from "../src/generation/program.ts";
import { createSampler, type Sampler } from "../src/generation/sampler.ts";
import {
  createGenerationSequence,
  type GenerationEvent,
  type GenerationGateTrace,
  type GenerationRequest,
  type GenerationRunPhase,
  type GenerationSequence,
  type GenerationStop,
  type GenerationStream,
} from "../src/generation/sequence.ts";
import {
  acceptDrafts,
  type DraftCycle,
  type DraftFace,
  planDraftLength,
  takeDrafts,
  verifyRowIndices,
} from "../src/generation/speculation.ts";
import type { SpeculationGateOptions } from "../src/generation/speculation-gate.ts";
import {
  drain,
  type FakeOptions,
  type FakeSession,
  fakeSession,
  hiddenMark,
  programOf,
  type RunCall,
  runShape,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

// ---- the fake target (successor function) and the wiring -----------------------------

/**
 * The fake target's successor function (the row's input token → the row's argmax).
 *
 * Its image is 2..14, so **0 / 1 / 15 never come out** — 0 is the pad-row value, and 15 can serve
 * as {@link REJECTED} (a draft that is always rejected).
 */
const SUCCESSOR: readonly number[] = Array.from(
  { length: VOCAB },
  (_unused, token) => ((token * 7 + 3) % 13) + 2,
);

/**
 * A token the target cannot choose (outside the image of `SUCCESSOR`) = a draft that is always
 * rejected.
 */
const REJECTED = 15;

/**
 * Logits that really give two choices on a temperature > 0 turn (first candidate 10 / runner-up 9.5
 * ≒ 62% : 38%).
 *
 * The default fake is unimodal ("10 for the targeted id, 0 for every other"), so raising the
 * temperature still collapses the draw to effectively one point — the RNG is consumed but **the
 * sequence does not move**, so it would not show that "speculative and non-speculative runs consume
 * the RNG in step". The runner-up id is taken outside the image of `SUCCESSOR` (2..14) (if it
 * collided with a row's first candidate, that row alone would go back to unimodal).
 */
const RUNNER_UP = 1;
const TWO_WAY: FakeOptions = { successor: SUCCESSOR, runnerUp: { id: RUNNER_UP, logit: 9.5 } };
/**
 * A seed whose draws pick both the first candidate and the runner-up (pinned by the step below
 * "giving a different sequence").
 */
const TWO_WAY_SEED = 7;

/** The turn's prompt (following `SUCCESSOR` from the trailing 2 gives 6, 8, 9, 3, 13, 5, …). */
const PROMPT: readonly number[] = [1, 2];

/** The number of draft steps (corresponds to the distribution's drafter). */
const K = 3;

/**
 * The speculation wiring: a bucket that holds verify's `k+1 = 4` rows is needed (chunkLength stays
 * as another shape).
 */
const SPEC_CHUNK_LENGTH = 8;
const SPEC_BUCKETS: readonly number[] = [4];

const specProgram = (fake: FakeSession, override: Partial<GenerationProgramSpec> = {}) =>
  programOf(fake, {
    chunkLength: SPEC_CHUNK_LENGTH,
    chunkBuckets: [...SPEC_BUCKETS],
    ...override,
  });

/** The `count` tokens after `token` that the target really emits (= drafts that all hit). */
const chainFrom = (token: number, count: number): number[] => {
  const chain: number[] = [];
  let current = token;
  for (let index = 0; index < count; index += 1) {
    current = SUCCESSOR[current];
    chain.push(current);
  }
  return chain;
};

// ---- the fake drafter (the borrower's surface, injected by DI) -------------------------

/**
 * The input of one cycle the drafter received (held as a copy — the record does not move if the
 * caller reuses it).
 */
type DraftRecord = {
  readonly token: number;
  readonly position: number;
  readonly hidden: readonly number[];
};

type FakeDrafter = ReturnType<typeof fakeDrafter>;

const fakeDrafter = (options: {
  /** The count emitted per run (default {@link K}). The caller uses only the first `k'` of them. */
  readonly steps?: number;
  /** The draft per cycle (`index` is the 0-based cycle number). */
  readonly draft: (cycle: DraftCycle, index: number) => readonly number[];
  /** Where to record for checking the dispose order (pushes `"drafter"`). */
  readonly disposeLog?: string[];
}) => {
  const steps = options.steps ?? K;
  const calls: DraftRecord[] = [];
  let disposals = 0;
  /**
   * The lender (tied by {@link watch}) — the condition under which the real runtime rejects a
   * borrower run is mirrored here.
   */
  let lender: FakeSession | undefined;
  const face: DraftFace = {
    steps,
    draft: (cycle) => {
      // MUST: in the real runtime a borrower run **also takes the lender's lease**, so a draft is
      // rejected while the lender still has an uncommitted deferred run (`acquireRun` in
      // `generation-context.ts`). An optimization that moves the draft from just before verify to
      // just after acceptance breaks this rule, but with a fake that ignores the lender nothing
      // would turn red without a GPU.
      assertEquals(
        lender?.pendingCommit(),
        undefined,
        "a draft was taken while the lender still had a pending (uncommitted verify) run",
      );
      const index = calls.length;
      calls.push({ token: cycle.token, position: cycle.position, hidden: [...cycle.hidden] });
      return Promise.resolve(Int32Array.from(options.draft(cycle, index)));
    },
    dispose: (): Promise<void> => {
      disposals += 1;
      options.disposeLog?.push("drafter");
      return Promise.resolve();
    },
  };
  return {
    face,
    calls,
    disposals: (): number => disposals,
    /**
     * Ties the lender (called by the side that builds the sequence — the borrower is created after
     * the lender).
     */
    watch: (fake: FakeSession): void => {
      lender = fake;
    },
  };
};

/** A drafter that always misses (forces 0 accepted). */
const wrongDrafter = (disposeLog?: string[]): FakeDrafter =>
  fakeDrafter({
    draft: () => Array.from({ length: K }, () => REJECTED),
    disposeLog,
  });

/** A drafter that copies the target's future as is (forces all accepted). */
const oracleDrafter = (steps = K): FakeDrafter =>
  fakeDrafter({ steps, draft: (cycle) => chainFrom(cycle.token, steps) });

/**
 * A drafter that hits only the first `accept` (partial acceptance — it always misses right after
 * the `accept`-th).
 */
const partialDrafter = (accept: number): FakeDrafter =>
  fakeDrafter({
    draft: (cycle) =>
      chainFrom(cycle.token, K).map((id, index) => (index < accept ? id : REJECTED)),
  });

// ---- run templates ---------------------------------------------------------------

const verifyCalls = (fake: FakeSession): RunCall[] =>
  fake.calls.filter((call) => call.commit === "deferred");

/**
 * The stop with the `speculation` field dropped (to compare with the non-speculative reference
 * run).
 */
const withoutSpeculation = (stop: GenerationStop): Record<string, unknown> => {
  const fields: Record<string, unknown> = { ...stop };
  delete fields.speculation;
  return fields;
};

const openSpeculative = async (options: {
  readonly drafter: FakeDrafter;
  readonly k?: number;
  readonly program?: Partial<GenerationProgramSpec>;
  readonly session?: FakeOptions;
  /**
   * How speculation is set up (default `"always"`).
   *
   * The checks in this file are **the contract without the gate** (1 cycle = draft + verify), so
   * the default is `"always"` — the self-accounting gate is checked separately by T16 with a fake
   * clock.
   */
  readonly policy?: "auto" | "always";
  /**
   * The gate's knobs (effective only with `policy: "auto"`).
   *
   * The default block is 16 cycles × 2 in a row, so a 20-token turn **never reaches a decision**.
   * Checks that look at the gate's wiring (accounting of dropped steps, the hidden-state
   * continuity, cycles not mixed into the observation) shrink the block, and the decisions
   * themselves are checked in `generation_gate_test.ts`.
   */
  readonly gate?: SpeculationGateOptions;
  /** The fake clock (read only with `policy: "auto"`). */
  readonly now?: () => number;
  readonly clock?: { readonly now: () => number; readonly afterRun: () => void };
  readonly onRun?: (phase: GenerationRunPhase) => void;
}) => {
  const fake = fakeSession({ successor: SUCCESSOR, ...options.session });
  options.drafter.watch(fake);
  const phases: GenerationRunPhase[] = [];
  const sequence = await createGenerationSequence({
    session: {
      ...fake.session,
      run: async (...args) => {
        const result = await fake.session.run(...args);
        if (args[2]?.commit === "deferred") options.clock?.afterRun();
        return result;
      },
    },
    program: specProgram(fake, options.program),
    speculative: {
      open: () => Promise.resolve(options.drafter.face),
      k: options.k,
      policy: options.policy ?? "always",
      ...(options.gate === undefined ? {} : { gate: options.gate }),
      ...((options.clock?.now ?? options.now) === undefined
        ? {}
        : { now: options.clock?.now ?? options.now }),
    },
    onRun: (phase) => {
      phases.push(phase);
      options.onRun?.(phase);
    },
  });
  return { fake, drafter: options.drafter, phases, sequence };
};

const runSpeculative = async (options: {
  readonly drafter: FakeDrafter;
  readonly request: GenerationRequest;
  readonly k?: number;
  readonly program?: Partial<GenerationProgramSpec>;
  readonly session?: FakeOptions;
  readonly policy?: "auto" | "always";
  readonly gate?: SpeculationGateOptions;
  readonly now?: () => number;
}) => {
  const opened = await openSpeculative(options);
  const drained = await drain(opened.sequence.generate(options.request));
  return { ...opened, ...drained };
};

/**
 * Builds a **non-speculative** sequence (no drafter injected) — the counterpart side that checks
 * the same invariants as speculation.
 */
const openPlain = async (
  program: Partial<GenerationProgramSpec> = {},
  session: FakeOptions = {},
) => {
  const fake = fakeSession({ successor: SUCCESSOR, ...session });
  const phases: GenerationRunPhase[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake, program),
    onRun: (phase) => {
      phases.push(phase);
    },
  });
  return { fake, phases, sequence };
};

/** The reference run of the same request **without speculation**. */
const runPlain = async (
  request: GenerationRequest,
  program: Partial<GenerationProgramSpec> = {},
  session: FakeOptions = {},
) => {
  const opened = await openPlain(program, session);
  const drained = await drain(opened.sequence.generate(request));
  return { ...opened, ...drained };
};

/**
 * Checks that a speculative run gives **the same output** as a non-speculative run (the core of
 * this contract).
 *
 * Compared are the event sequence (token ids and positions, prefill progress) and the stop (every
 * field except `speculation`).
 */
const assertMatchesPlain = async (
  request: GenerationRequest,
  speculated: { readonly events: readonly GenerationEvent[]; readonly stop: GenerationStop },
  program: Partial<GenerationProgramSpec> = {},
  session: FakeOptions = {},
): Promise<void> => {
  const plain = await runPlain(request, program, session);
  assertEquals(
    speculated.events,
    plain.events,
    "the speculative token sequence / positions differ from the non-speculative run",
  );
  assertEquals(
    withoutSpeculation(speculated.stop),
    withoutSpeculation(plain.stop),
    "the speculative stop differs from the non-speculative run",
  );
  assertEquals(
    Object.hasOwn(plain.stop, "speculation"),
    false,
    "the non-speculative sequence's stop has a speculation field",
  );
};

// ---- T1–T3: acceptance counts and identical output ------------------------------------

test("T1 forced rejection: drafts that all miss still give the non-speculative sequence (one frontier committed per cycle)", async () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 5 };
  const run = await runSpeculative({ drafter: wrongDrafter(), request });
  await assertMatchesPlain(request, run);

  // A 0-accepted cycle commits only b′ = it degenerates to one non-speculative decode step.
  assertEquals(run.fake.commits, [1, 1, 1, 1], "a 0-accepted cycle committed more than one row");
  assertEquals(run.stop.speculation, {
    cycles: 4,
    draftRuns: 3,
    drafted: 6,
    accepted: 0,
    delivered: 4,
    acceptedHistogram: [4, 0, 0, 0],
  });
  // The remaining budget (`k' = min(k, remaining − 1)`) shrinks the count, and the last cycle takes
  // no draft.
  assertEquals(run.drafter.calls.length, 3);
  assertEquals(
    verifyCalls(run.fake).map((call) => call.queryLength),
    [4, 3, 2, 1],
  );
});

test("T2 all accepted: drafts that copy the future as is commit k+1 tokens per verify", async () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await runSpeculative({ drafter: oracleDrafter(), request });
  await assertMatchesPlain(request, run);

  assertEquals(run.fake.commits, [4, 4], "an all-accepted cycle commits k+1 rows");
  assertEquals(run.stop.speculation, {
    cycles: 2,
    draftRuns: 2,
    drafted: 6,
    accepted: 6,
    delivered: 8,
    acceptedHistogram: [0, 0, 0, 2],
  });
  // The lender issues 3 runs, prefill 1 + verify 2 (non-speculative: 9 runs, prefill 1 + decode 8).
  assertEquals(run.fake.calls.length, 3);
  assertEquals((await runPlain(request)).fake.calls.length, 9);
});

describe("T3 partial acceptance: the position where the prefix match stops is the accepted count (a = 1 / 2)", () => {
  it("a = 1 (only the first hits)", async () => {
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 7 };
    const run = await runSpeculative({ drafter: partialDrafter(1), request });
    await assertMatchesPlain(request, run);

    // Each commit is the 2 tokens `[d₁, b′]` (the rejected rows d₂ / d₃ do not enter the
    // conversation).
    assertEquals(run.fake.commits, [2, 2, 2]);
    assertEquals(run.stop.speculation, {
      cycles: 3,
      draftRuns: 3,
      drafted: 7,
      accepted: 3,
      delivered: 6,
      acceptedHistogram: [0, 3, 0, 0],
    });
  });

  it("a = 2 (the first two hit)", async () => {
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 7 };
    const run = await runSpeculative({ drafter: partialDrafter(2), request });
    await assertMatchesPlain(request, run);

    assertEquals(run.fake.commits, [3, 3]);
    assertEquals(run.stop.speculation, {
      cycles: 2,
      draftRuns: 2,
      drafted: 5,
      accepted: 4,
      delivered: 6,
      acceptedHistogram: [0, 0, 2, 0],
    });
  });
});

describe("speculative acceptance masks the wiring's suppressed set", () => {
  // The raw chain from the prompt is 6, 8, 9, …; 8 is suppressed, so the masked target turns to the
  // runner-up (id 1, logit 9.5) there. The oracle drafter proposes the **raw** chain, so a verify
  // row that accepted against unmasked logits would take 8.
  const SUPPRESSED = 8;
  const program: Partial<GenerationProgramSpec> = { suppressTokens: [SUPPRESSED] };

  for (const temperature of [0, 1]) {
    it(`temperature ${temperature}: the same tokens as the masked plain run, never the suppressed id`, async () => {
      // Control: a seed whose unmasked plain run takes 8. Masking changes no row before the first
      // 8, so the masked run reaches the same row (after 6) and must take something else there.
      const requestOf = (seed: number): GenerationRequest => ({
        prompt: PROMPT,
        maxNewTokens: 9,
        sampler: { temperature, seed },
      });
      let seed: number | undefined;
      for (let candidate = 0; candidate < 32 && seed === undefined; candidate += 1) {
        const control = await runPlain(requestOf(candidate), {}, TWO_WAY);
        if (tokenIds(control.events).includes(SUPPRESSED)) seed = candidate;
      }
      assert(seed !== undefined, "control: no seed of 0..31 takes 8 unmasked");
      const request = requestOf(seed);

      const run = await runSpeculative({
        drafter: oracleDrafter(),
        request,
        program,
        session: TWO_WAY,
      });
      await assertMatchesPlain(request, run, program, TWO_WAY);
      assert(!tokenIds(run.events).includes(SUPPRESSED), tokenIds(run.events).join(","));
      // The drafts carried the suppressed id and were rejected at that row.
      assert(
        run.stop.speculation !== undefined &&
          run.stop.speculation.accepted < run.stop.speculation.drafted,
      );
    });
  }
});

describe("T3′ temperature > 0: speculation is set up even on turns that draw, and the sequence matches the non-speculative one exactly", () => {
  // Acceptance calls `sampler.next` per row with the same logits, the same history and in the same
  // order as "the draw non-speculative decode makes at the same position", **once per committed
  // token**. The RNG consumption lines up one to one with the non-speculative run, so the token
  // sequence does not change at temperature > 0 either (speculative sampling with a one-hot draft).
  const request = (seed: number): GenerationRequest => ({
    prompt: PROMPT,
    maxNewTokens: 9,
    sampler: { temperature: 1, topK: 2, seed },
  });

  it("the speculative / non-speculative sequences, positions and stops match, and the accounting is present", async () => {
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED),
      session: TWO_WAY,
    });
    await assertMatchesPlain(request(TWO_WAY_SEED), run, {}, TWO_WAY);

    const speculation = run.stop.speculation;
    assert(speculation !== undefined, "a temperature > 0 turn carries no speculation field");
    assert(speculation.cycles > 0, "not a single verify run happened");
    assertEquals(withoutSpeculation(run.stop), { reason: "max-tokens", tokens: 9 });
    // It passes through both cycles where the draft (= the temperature-0 chain) hits and cycles
    // where it misses = the acceptance decision branches on the draw's result (it is not stuck at
    // all accepted / all rejected).
    assert(
      speculation.accepted > 0 && speculation.accepted < speculation.drafted,
      `acceptance is stuck at all or 0: ${JSON.stringify(speculation)}`,
    );
  });

  it("the draw really runs (it gives a different sequence from temperature 0)", async () => {
    const sampled = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED),
      session: TWO_WAY,
    });
    const greedy = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens: 9 },
      session: TWO_WAY,
    });
    // The temperature-0 sequence is the successor function itself (the runner-up's 9.5 does not
    // exceed the first candidate's 10).
    assertEquals(tokenIds(greedy.events), chainFrom(PROMPT[PROMPT.length - 1], 9));
    assert(
      tokenIds(sampled.events).join() !== tokenIds(greedy.events).join(),
      `only the argmax sequence at temperature 1 (the RNG is not consumed): ${tokenIds(
        sampled.events,
      ).join(",")}`,
    );
  });

  it("the sequence stays the same with the plain steps the self-accounting gate inserts", async () => {
    // Temperature > 0 combined with the gate (a losing wall time fed through the fake clock). The
    // decode-shaped steps the gate drops to also draw exactly as "the draw non-speculative decode
    // makes at the same position" (the same logits, the same history, once in the same order), so
    // the RNG consumption stays in step even when speculative cycles and plain steps mix.
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      session: TWO_WAY,
      clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? 100 : 10)),
    });
    // The block (2 cycles) has to fill, so only this step gets a larger budget.
    const budgeted: GenerationRequest = { ...request(TWO_WAY_SEED), maxNewTokens: 24 };
    const gated = { ...opened, ...(await drain(opened.sequence.generate(budgeted))) };
    // The gate has really **switched** (do not read the single re-measurement move of `W1` as "it
    // dropped").
    assertEquals(
      gated.stop.speculation?.switches,
      1,
      `the gate did not switch to plain: ${JSON.stringify(gated.stop.speculation)}`,
    );
    await assertMatchesPlain(budgeted, gated, {}, TWO_WAY);

    // It also matches `always` (no gate) = the gate changes only the speed.
    const always = await runSpeculative({
      drafter: oracleDrafter(),
      request: budgeted,
      session: TWO_WAY,
    });
    assertEquals(
      tokenIds(gated.events),
      tokenIds(always.events),
      "the gated and always sequences differ",
    );
  });

  it("a different seed gives a different sequence (the RNG state is tied to one generation)", async () => {
    const first = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED),
      session: TWO_WAY,
    });
    const second = await runSpeculative({
      drafter: oracleDrafter(),
      request: request(TWO_WAY_SEED + 1),
      session: TWO_WAY,
    });
    await assertMatchesPlain(request(TWO_WAY_SEED + 1), second, {}, TWO_WAY);
    assert(
      tokenIds(first.events).join() !== tokenIds(second.events).join(),
      "changing the seed gives the same sequence",
    );
  });
});

// ---- T4 / T5: stop tokens ------------------------------------------------------------

test("T4 a stop token in the middle of the draft: acceptance ends there and the rows after it are not committed", async () => {
  // The chain is 6 → 8 → 9 → 3 → 13. The stop token 9 falls on d₂ of cycle 1.
  const stopTokens = [9];
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request,
    program: { stopTokens },
  });
  await assertMatchesPlain(request, run, { stopTokens });

  // The stop token is not body text, so it is not emitted as a `token` event, but it counts as one.
  assertEquals(tokenIds(run.events), [6, 8]);
  assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 9, tokens: 3 });
  // Acceptance goes up to the stop token 9 (the d₃ row is not even drawn) = the committed sequence
  // `[8, 9]` itself is what is delivered.
  const speculation = run.stop.speculation;
  assertEquals(speculation, {
    cycles: 1,
    draftRuns: 1,
    drafted: 3,
    accepted: 2,
    delivered: 2,
    acceptedHistogram: [0, 0, 1, 0],
  });
  // Why the source of truth for token/cycle is `delivered / cycles` (this cycle does not commit
  // `1 + a` tokens).
  assert(speculation !== undefined);
  assertEquals(
    speculation.accepted + speculation.cycles,
    3,
    "the old numerator would be too large by 1",
  );
  assertEquals(run.fake.commits, [2], "committed the candidates after the stop token too");
  assertEquals(
    verifyCalls(run.fake)[0].queryLength,
    4,
    "k+1 rows were written to the physical ring",
  );

  // The stop token stays in the conversation = it is prepended to the next turn's prefill (the same
  // cleanup as non-speculative).
  await drain(run.sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  const next = run.fake.calls[run.fake.calls.length - 1];
  assertEquals(next.ids, [9, 4, 0, 0]);
  assertEquals(next.positions, [4, 5, 0, 0]);
});

test("T4′ rows after the stop token are not drawn (it does not fail on logits the non-speculative run never touches)", async () => {
  // Fault injection: mix NaN into **row 2 only** of the verify (call 1). Row 2 is "the row after
  // the stop token 9 was committed" = a position where the non-speculative run issues no run at
  // all. An implementation that does not end acceptance at the stop token calls `sampler.next` here
  // too, so `assertNoNaN` throws (the shape that really happens with the per-row NaN poisoning of
  // an out-of-range gather — "fails only when speculating" = a breach of the contract that the
  // output does not change).
  const stopTokens = [9];
  const session: FakeOptions = { successor: SUCCESSOR, nanAt: 1, nanRow: 2 };
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request,
    program: { stopTokens },
    session,
  });
  // It closes with the same sequence as the non-speculative reference run (decode is one row, so
  // there is no row 2 and it does not hit the poisoning).
  await assertMatchesPlain(request, run, { stopTokens }, session);

  assertEquals(tokenIds(run.events), [6, 8]);
  assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 9, tokens: 3 });
  assertEquals(run.fake.commits, [2]);
});

describe("T5 a stop token as b′: the accepted drafts are delivered and it closes on the new frontier", () => {
  it("b′ is a stop token after one accepted draft", async () => {
    // 6 → 8 (accepted) → b′ = 9 (stop). No cutoff happens; the committed sequence closes on the
    // stop as is.
    const stopTokens = [9];
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
    const run = await runSpeculative({
      drafter: partialDrafter(1),
      request,
      program: { stopTokens },
    });
    await assertMatchesPlain(request, run, { stopTokens });

    assertEquals(tokenIds(run.events), [6, 8]);
    assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 9, tokens: 3 });
    assertEquals(run.stop.speculation, {
      cycles: 1,
      draftRuns: 1,
      drafted: 3,
      accepted: 1,
      delivered: 2,
      acceptedHistogram: [0, 1, 0, 0],
    });
    assertEquals(run.fake.commits, [2]);
  });

  it("b′ is a stop token with 0 accepted (one committed token)", async () => {
    const stopTokens = [8];
    const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
    const run = await runSpeculative({ drafter: wrongDrafter(), request, program: { stopTokens } });
    await assertMatchesPlain(request, run, { stopTokens });

    assertEquals(tokenIds(run.events), [6]);
    assertEquals(withoutSpeculation(run.stop), { reason: "eos", token: 8, tokens: 2 });
    assertEquals(run.fake.commits, [1]);
  });
});

// ---- T6: the end of the budget ---------------------------------------------------------

test("T6 end of the budget: a cycle with one token left takes no draft and closes with a one-row, decode-shaped verify", async () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 6 };
  const run = await runSpeculative({ drafter: oracleDrafter(), request });
  await assertMatchesPlain(request, run);

  // Cycle 1 commits 4 (1 left) → cycle 2 has `k' = 0`.
  assertEquals(run.fake.commits, [4, 1]);
  assertEquals(
    run.drafter.calls.length,
    1,
    "a draft run was issued on the cycle with one token left",
  );
  const verifies = verifyCalls(run.fake);
  assertEquals(
    verifies.map((call) => call.queryLength),
    [4, 1],
  );
  assertEquals(
    verifies.map((call) => call.lastRows),
    [[0, 1, 2, 3], [0]],
  );
  assertEquals(
    verifies.map((call) => call.idsShape),
    [
      [1, 4],
      [1, 1],
    ],
    "k'=0 is the decode shape",
  );
  assertEquals(
    verifies.map((call) => call.commit),
    ["deferred", "deferred"],
  );
  // The tokens of `max-tokens` are exactly the requested number (neither more nor fewer).
  assertEquals(withoutSpeculation(run.stop), { reason: "max-tokens", tokens: 6 });
  // A `k' = 0` cycle takes no draft, so it accepts 0 = it goes into index 0 of the histogram.
  assertEquals(run.stop.speculation, {
    cycles: 2,
    draftRuns: 1,
    drafted: 3,
    accepted: 3,
    delivered: 5,
    acceptedHistogram: [1, 0, 0, 1],
  });
});

// ---- T7: interruption, exception, break -------------------------------------------------

test("T7 interruption: stopping before the draft adds neither a run nor a draft", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped before the draft");
  const opened = await openSpeculative({ drafter: oracleDrafter() });
  const stream = opened.sequence.generate({
    prompt: PROMPT,
    maxNewTokens: 9,
    signal: controller.signal,
  });

  const seen: number[] = [];
  let caught: unknown;
  try {
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      seen.push(event.id);
      controller.abort(reason);
    }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `the interruption is wrapped: ${String(caught)}`);
  assertEquals(seen, [6]);
  assertEquals(await stream.done, {
    reason: "aborted",
    tokens: 1,
    speculation: {
      cycles: 0,
      draftRuns: 0,
      drafted: 0,
      accepted: 0,
      delivered: 0,
      acceptedHistogram: [0, 0, 0, 0],
    },
  });
  assertEquals(opened.drafter.calls.length, 0);
  assertEquals(opened.fake.calls.length, 1, "a run after the prefill was issued");
  assertEquals(opened.fake.commits, []);
});

test("T7 interruption: stopping after the draft and before the verify leaves no pending run", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped between the draft and the verify");
  const drafter = fakeDrafter({
    draft: (cycle) => {
      controller.abort(reason);
      return chainFrom(cycle.token, K);
    },
  });
  const opened = await openSpeculative({ drafter });
  const stream = opened.sequence.generate({
    prompt: PROMPT,
    maxNewTokens: 9,
    signal: controller.signal,
  });

  let caught: unknown;
  try {
    for await (const _event of stream) {
      /* drain until the interruption */
    }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `the interruption is wrapped: ${String(caught)}`);
  assertEquals(await stream.done, {
    reason: "aborted",
    tokens: 1,
    speculation: {
      cycles: 0,
      draftRuns: 1,
      drafted: 3,
      accepted: 0,
      delivered: 0,
      acceptedHistogram: [0, 0, 0, 0],
    },
  });
  assertEquals(opened.drafter.calls.length, 1, "one draft ran");
  assertEquals(
    opened.fake.calls.length,
    1,
    "the verify was issued (the interruption takes effect just before a run is issued)",
  );
  assertEquals(opened.fake.pendingCommit(), undefined, "a pending run remains");
  assertEquals(opened.fake.commits, []);
});

test("T7 interruption: stopping after the verify still delivers and commits that cycle's committed tokens", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped in the middle of delivery");
  const opened = await openSpeculative({ drafter: oracleDrafter() });
  const stream = opened.sequence.generate({
    prompt: PROMPT,
    maxNewTokens: 9,
    signal: controller.signal,
  });

  const seen: number[] = [];
  let caught: unknown;
  try {
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      seen.push(event.id);
      // Interrupt on receiving the first token of cycle 1 (the rest is a synchronous section, so it
      // is all delivered).
      if (seen.length === 2) controller.abort(reason);
    }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `the interruption is wrapped: ${String(caught)}`);
  // The interruption takes effect at **a stage boundary** (the head of the next cycle) — a running
  // cycle is not dropped midway.
  assertEquals(seen, [6, 8, 9, 3, 13]);
  assertEquals(await stream.done, {
    reason: "aborted",
    tokens: 5,
    speculation: {
      cycles: 1,
      draftRuns: 1,
      drafted: 3,
      accepted: 3,
      delivered: 4,
      acceptedHistogram: [0, 0, 0, 1],
    },
  });
  assertEquals(opened.fake.commits, [4], "the delivered tokens are not committed");
  assertEquals(opened.fake.pendingCommit(), undefined);
  assertEquals(opened.fake.calls.length, 2);
  // One report per run (no second verify is issued because of the interruption).
  assertEquals(runShape(opened.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
  ]);
});

test("T7′ exception recovery: a failure on the verify's return leaves no pending run, and the same sequence passes the next turn", async () => {
  // Fault injection: only the first verify (call 1) returns a row count different from the one
  // requested. The reader's shape check fails in the synchronous section "from the deferred run's
  // return to delivery", which is the only window that leaves a pending run.
  const opened = await openSpeculative({
    drafter: oracleDrafter(),
    session: { successor: SUCCESSOR, logitsRows: 2, logitsRowsAt: 1 },
  });
  await assertRejects(
    () => drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 })),
    Error,
    `verify@1: 'logits' has shape [1,2,${VOCAB}], not [1,4,${VOCAB}]`,
  );
  // Folded with `commit(0)`: b stays unfed and the rejected rows are overwritten by the next run.
  assertEquals(opened.fake.commits, [0], "the exception path did not go through commit(0)");
  assertEquals(
    opened.fake.pendingCommit(),
    undefined,
    "a pending run remains (the next run would fail)",
  );
  assertEquals(
    opened.fake.pastLength(),
    PROMPT.length,
    "the logical length advanced by the verify",
  );
  // A run that failed before its output was read is not reported either (`onRun` is the hook for
  // "runs that were read").
  assertEquals(runShape(opened.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
  ]);

  // The next turn passes on the same sequence (the frontier 6 is prepended and no token is lost).
  const next = await drain(opened.sequence.generate({ prompt: [4], maxNewTokens: 2 }));
  assertEquals(opened.fake.calls[2].ids, [6, 4, 0, 0]);
  assertEquals(opened.fake.calls[2].positions, [2, 3, 0, 0]);
  assertEquals(tokenIds(next.events), [SUCCESSOR[4], SUCCESSOR[SUCCESSOR[4]]]);
  assertEquals(withoutSpeculation(next.stop), { reason: "max-tokens", tokens: 2 });
});

test("T7″ break midway: up to the delivered tokens is committed, and the last of them heads the next turn", async () => {
  const opened = await openSpeculative({ drafter: oracleDrafter() });
  const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });

  const seen: number[] = [];
  for await (const event of stream) {
    if (event.kind !== "token") continue;
    seen.push(event.id);
    // Stop at the second token of cycle 1 (the third overall) = enters finally through `return()`.
    if (seen.length === 3) break;
  }
  assertEquals(seen, [6, 8, 9]);
  // `delivered` adds the cycle's count when acceptance is decided (at the same point as
  // `accepted`), so on this turn, whose delivery closed midway on `break`, it is 4 — more than the
  // 2 the consumer received.
  assertEquals(await stream.done, {
    reason: "closed",
    tokens: 3,
    speculation: {
      cycles: 1,
      draftRuns: 1,
      drafted: 3,
      accepted: 3,
      delivered: 4,
      acceptedHistogram: [0, 0, 0, 1],
    },
  });
  // The undelivered committed candidates (3 / 13) do not enter the conversation — the consumer's
  // transcript and the conversation agree.
  assertEquals(opened.fake.commits, [2]);
  assertEquals(opened.fake.pendingCommit(), undefined);
  // Even when delivery closes midway, the verify run that ran is reported exactly once.
  assertEquals(runShape(opened.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
  ]);

  await drain(opened.sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  const next = opened.fake.calls[opened.fake.calls.length - 1];
  assertEquals(
    next.ids,
    [9, 4, 0, 0],
    "the frontier at the break position does not head the next turn",
  );
  assertEquals(next.positions, [4, 5, 0, 0]);
});

// ---- T8: checks (at sequence creation, sampler) ---------------------------------------

describe("T8 checks: a speculation spec fails at sequence creation, and every opened surface is torn down", () => {
  /**
   * Checks in one place that it is rejected at creation + torn down in the order borrower → lender.
   */
  const assertRejected = async (
    options: {
      readonly k?: number;
      readonly program?: Partial<GenerationProgramSpec>;
      readonly session?: FakeOptions;
      readonly steps?: number;
    },
    message: string,
  ): Promise<void> => {
    const disposeLog: string[] = [];
    const fake = fakeSession({ successor: SUCCESSOR, disposeLog, ...options.session });
    const drafter = fakeDrafter({
      steps: options.steps ?? K,
      draft: () => chainFrom(1, K),
      disposeLog,
    });
    await assertRejects(
      () =>
        createGenerationSequence({
          session: fake.session,
          program: specProgram(fake, options.program),
          speculative: {
            open: () => Promise.resolve(drafter.face),
            k: options.k,
            policy: "always",
          },
        }),
      Error,
      message,
    );
    assertEquals(drafter.disposals(), 1, "the drafter's surface was not torn down");
    assertEquals(fake.disposals(), 1, "the lender context was not torn down");
    assertEquals(disposeLog, ["drafter", "context"], "the teardown order is not borrower → lender");
    assertEquals(fake.calls.length, 0);
  };

  it("k + 1 exceeds the sliding ring's slack", async () => {
    await assertRejected(
      { k: K, session: { slidingSlack: K } },
      `speculative.k ${K} does not fit in the sliding ring's slack ${K}`,
    );
  });

  it("k exceeds the drafter's step count", async () => {
    await assertRejected(
      { k: K + 1 },
      `speculative.k ${K + 1} is outside 1..${K} (the drafter's step count)`,
    );
  });

  it("k is below 1", async () => {
    await assertRejected({ k: 0 }, `speculative.k 0 is outside 1..${K} (the drafter's step count)`);
  });

  it("the drafter's declared step count is not a positive integer", async () => {
    // Omitting `k` gives `k = steps`, so an implementation that does not check the step count lets
    // the `k` check pass (every comparison with NaN is false, and `1.5 > 1.5` is false). The
    // message (on the step-count side) pins that the step-count check stands **before** the `k`
    // check.
    await assertRejected({ steps: Number.NaN }, "Drafter step count NaN is not a positive integer");
    await assertRejected({ steps: 0 }, "Drafter step count 0 is not a positive integer");
    await assertRejected({ steps: 1.5 }, "Drafter step count 1.5 is not a positive integer");
  });

  it("the k + 1 rows exceed chunkLength", async () => {
    await assertRejected(
      { k: K, program: { chunkLength: K, chunkBuckets: [] } },
      `The verify of speculative.k ${K} (${K + 1} rows) exceeds chunkLength ${K}`,
    );
  });

  it("no bucket holds the k + 1 rows", async () => {
    await assertRejected(
      { k: K, program: { chunkBuckets: [2] } },
      `No bucket holds the ${K + 1} verify rows (chunkBuckets [2])`,
    );
  });

  it("a context without a sliding slot does not go through the slack check", async () => {
    // With `slidingSlack` `undefined` (no sliding), there is no past KV for rejected rows to
    // overwrite either.
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens: 5 },
      session: { successor: SUCCESSOR, slidingSlack: undefined },
    });
    assertEquals(run.fake.commits, [4]);
  });

  it('passing gate knobs with policy "always" fails before open, and the lender is torn down', async () => {
    const fake = fakeSession({ successor: SUCCESSOR });
    let caught: unknown;
    try {
      await createGenerationSequence({
        session: fake.session,
        program: specProgram(fake),
        speculative: {
          open: () => Promise.reject(new Error("must not reach open")),
          policy: "always",
          gate: { burstAbort: 10 },
        },
      });
    } catch (error) {
      caught = error;
    }
    assert(caught instanceof Error, `did not fail: ${String(caught)}`);
    assertStringIncludes(caught.message, '"always" has no gate');
    assertEquals(
      fake.disposals(),
      1,
      "the lender leaked along with the borrower that failed the check",
    );
  });

  it("when open throws, the lender context is torn down and the error passes through as is", async () => {
    const fake = fakeSession({ successor: SUCCESSOR });
    const reason = new Error("the drafter cannot be opened");
    let caught: unknown;
    try {
      await createGenerationSequence({
        session: fake.session,
        program: specProgram(fake),
        speculative: { open: () => Promise.reject(reason), policy: "always" },
      });
    } catch (error) {
      caught = error;
    }
    assert(caught === reason, `the open failure is wrapped: ${String(caught)}`);
    assertEquals(
      fake.disposals(),
      1,
      "the lender leaked along with the borrower that could not be opened",
    );
  });
});

test("T8 checks: open is called right after the context is allocated, with the lender context itself", async () => {
  const fake = fakeSession({ successor: SUCCESSOR });
  const drafter = oracleDrafter();
  const seen: unknown[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake),
    speculative: {
      open: (context) => {
        seen.push(context);
        // The borrower is opened right after the context is allocated (no run has been issued).
        assertEquals(fake.specs.length, 1);
        assertEquals(fake.calls.length, 0);
        return Promise.resolve(drafter.face);
      },
      policy: "always",
    },
  });
  assertEquals(seen.length, 1);
  assert(seen[0] === fake.context, "open received something other than the lender context");

  // Torn down in the order borrower → lender (the `dispose` chain).
  const disposeLog: string[] = [];
  const logged = fakeSession({ successor: SUCCESSOR, disposeLog });
  const loggedDrafter = fakeDrafter({ draft: () => chainFrom(1, K), disposeLog });
  const disposable = await createGenerationSequence({
    session: logged.session,
    program: specProgram(logged),
    speculative: { open: () => Promise.resolve(loggedDrafter.face), policy: "always" },
  });
  await disposable.dispose();
  assertEquals(disposeLog, ["drafter", "context"]);
  await sequence.dispose();
});

test("T8 checks: the stop of a sequence without speculative has no speculation field", async () => {
  const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 3 });
  assertEquals(Object.hasOwn(plain.stop, "speculation"), false);
  assertEquals(withoutSpeculation(plain.stop), { reason: "max-tokens", tokens: 3 });
});

// ---- T9: onRun hook ------------------------------------------------------------

test("T9 onRun: speculation is prefill → (draft, verify) × cycle (cycles start at 1)", async () => {
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt: PROMPT, maxNewTokens: 9 },
  });
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
    { kind: "draft", cycle: 2 },
    { kind: "verify", cycle: 2, rows: 4, accepted: 3 },
  ]);
});

test("T9 onRun: a k' = 0 cycle does not report a draft and reports only a one-row verify", async () => {
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt: PROMPT, maxNewTokens: 6 },
  });
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 3 },
    { kind: "verify", cycle: 2, rows: 1, accepted: 0 },
  ]);
});

test("T9 onRun: partial acceptance's accepted, and the progress of a prefill split into chunks", async () => {
  const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  const run = await runSpeculative({
    drafter: partialDrafter(1),
    request: { prompt, maxNewTokens: 3 },
  });
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 2 },
    { kind: "prefill", chunk: 2, chunks: 2 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 2, accepted: 1 },
  ]);
});

test("T9 onRun: non-speculative is prefill → decode × (tokens − 1) (steps start at 1)", async () => {
  const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 3 });
  assertEquals(runShape(plain.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "decode", step: 1 },
    { kind: "decode", step: 2 },
  ]);
});

test("T9 onRun: one report per run — the verify of a cycle closed by a stop is reported too", async () => {
  // `onRun` is the hook that counts runs (the denominator of diagnostics and tok/s). Dropping the
  // verify of a cycle closed by a stop token would remove that one run from the observation
  // (non-speculative decode reports the step that stopped as well).
  const stopTokens = [9];
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt: PROMPT, maxNewTokens: 9 },
    program: { stopTokens },
  });
  assertEquals(run.fake.calls.length, 2, "the runs were prefill 1 + verify 1");
  // Acceptance ends at the stop token (d₂ = 9), so `accepted` is 2 (the number of delivered
  // drafts).
  assertEquals(runShape(run.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "draft", cycle: 1 },
    { kind: "verify", cycle: 1, rows: 4, accepted: 2 },
  ]);

  // The counterpart (non-speculative reports the decode run that stopped as one report too).
  const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 9 }, { stopTokens });
  assertEquals(runShape(plain.phases), [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "decode", step: 1 },
    { kind: "decode", step: 2 },
  ]);
});

test("T9 onRun: a sequence without the hook gives the same sequence (the observation hook does not change behaviour)", async () => {
  const fake = fakeSession({ successor: SUCCESSOR });
  const drafter = oracleDrafter();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake),
    speculative: { open: () => Promise.resolve(drafter.face), policy: "always" },
  });
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 9 };
  const run = await drain(sequence.generate(request));
  await assertMatchesPlain(request, run);
  assertEquals(fake.commits, [4, 4]);
});

// ---- T10: speculation accounting -------------------------------------------------------

describe("T10 accounting: the histogram has length k+1, its index is the accepted count, and its sums agree with cycles / accepted", () => {
  const tally = async (k: number, drafter: FakeDrafter, maxNewTokens: number) => {
    const run = await runSpeculative({
      drafter,
      k,
      request: { prompt: PROMPT, maxNewTokens },
    });
    const speculation = run.stop.speculation;
    assert(speculation !== undefined, "a speculative turn carries no speculation field");
    return speculation;
  };

  it("k = 3 (the same as the drafter's step count)", async () => {
    const speculation = await tally(K, partialDrafter(2), 7);
    assertEquals(speculation.acceptedHistogram.length, K + 1);
    assertEquals(speculation.acceptedHistogram, [0, 0, 2, 0]);
    assertEquals(
      speculation.acceptedHistogram.reduce((sum, count) => sum + count, 0),
      speculation.cycles,
      "the histogram total does not match cycles",
    );
    assertEquals(
      speculation.acceptedHistogram.reduce((sum, count, accepted) => sum + count * accepted, 0),
      speculation.accepted,
      "the histogram's weighted sum does not match accepted",
    );
    // On a turn without any cycle cut off by a stop token, `delivered = accepted + cycles` (every
    // cycle commits `1 + a` tokens). A cycle cut off by a stop breaks this.
    assertEquals(speculation.delivered, speculation.accepted + speculation.cycles);
  });

  it("k = 2 (shorter than the step count)", async () => {
    const speculation = await tally(2, oracleDrafter(), 7);
    // Each cycle commits 3 → 2 cycles (1 + 3 + 3 = 7).
    assertEquals(speculation.acceptedHistogram.length, 3);
    assertEquals(speculation, {
      cycles: 2,
      draftRuns: 2,
      drafted: 4,
      accepted: 4,
      delivered: 6,
      acceptedHistogram: [0, 0, 2],
    });
  });

  it("k = 1 (1 draft = 2 verify rows)", async () => {
    const speculation = await tally(1, wrongDrafter(), 4);
    assertEquals(speculation, {
      cycles: 3,
      draftRuns: 2,
      drafted: 2,
      accepted: 0,
      delivered: 3,
      acceptedHistogram: [3, 0],
    });
  });
});

// ---- T11: the hidden-state copy --------------------------------------------------------

test("T11 hidden: what reaches the next cycle's drafter is the hidden state of the row that emitted b′ (row a)", async () => {
  // Cycles with one accepted draft follow one another, so the right answer is **row 1**, neither
  // row 0 nor row k.
  const run = await runSpeculative({
    drafter: partialDrafter(1),
    request: { prompt: PROMPT, maxNewTokens: 7 },
  });
  assertEquals(run.drafter.calls.length, 3);

  // Cycle 1: the hidden state of the prefill's (run 0) last valid row (index 1).
  assertEquals(run.drafter.calls[0], {
    token: 6,
    position: 2,
    hidden: [hiddenMark(0, 1), 2, 0, 0],
  });
  // Cycle 2: row 1 of the verify (run 1) = the row that emitted b′ 9 (row 0 would give 101, row 3
  // 104).
  assertEquals(run.drafter.calls[1], {
    token: 9,
    position: 4,
    hidden: [hiddenMark(1, 1), 8, 0, 0],
  });
  // Cycle 3: row 1 of the verify (run 2).
  assertEquals(run.drafter.calls[2], {
    token: 13,
    position: 6,
    hidden: [hiddenMark(2, 1), 3, 0, 0],
  });
});

test("T11 hidden: the first cycle's hidden state is row 0 of the prefill's **last chunk**", async () => {
  // A prompt that splits into 2 chunks (9 tokens / chunkLength 8) — an implementation that grabs
  // the first chunk's hidden state breaks here (the last chunk has one valid row = the decode
  // shape, so row 0).
  const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  const run = await runSpeculative({
    drafter: oracleDrafter(),
    request: { prompt, maxNewTokens: 3 },
  });
  assertEquals(run.fake.calls.length, 3, "2 prefills + 1 verify");
  assertEquals(run.drafter.calls, [
    {
      token: SUCCESSOR[9],
      position: prompt.length,
      hidden: [hiddenMark(1, 0), 9, 0, 0],
    },
  ]);
});

// ---- T12 / T13: the draft range check and last_row padding ------------------------------

describe("T12 range check: an out-of-vocabulary or short draft fails loudly before the verify is issued", () => {
  const assertDraftRejected = async (
    draft: () => readonly number[],
    message: string,
  ): Promise<void> => {
    const opened = await openSpeculative({ drafter: fakeDrafter({ draft }) });
    await assertRejects(
      () => drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 })),
      Error,
      message,
    );
    // A draft is a value that enters the verify's `input_ids` and the history, so it fails at the
    // same point as the prompt.
    assertEquals(opened.fake.calls.length, 1, "a verify was issued with an out-of-range draft");
    assertEquals(opened.fake.commits, []);
    assertEquals(opened.fake.pendingCommit(), undefined);
    // The draft run itself **has completed** (the borrower returned its drafts in full), so it
    // reports once. What fails is the range check of the return value, which is a different point
    // from the verify's shape check (T7′ — that fails before the output is read, so it does not
    // report).
    assertEquals(runShape(opened.phases), [
      { kind: "prefill", chunk: 1, chunks: 1 },
      { kind: "draft", cycle: 1 },
    ]);
  };

  it("above the top of the vocabulary", async () => {
    await assertDraftRejected(
      () => [VOCAB, 8, 9],
      `draft[0] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`,
    );
  });

  it("a negative id", async () => {
    await assertDraftRejected(
      () => [8, -1, 9],
      `draft[1] -1 is outside the vocabulary 0..${VOCAB - 1}`,
    );
  });

  it("too few drafts", async () => {
    await assertDraftRejected(() => [8, 9], "Only 2 drafts (3 needed)");
  });
});

describe("T13 last_row: R stays k+1 when k' shrinks (padded with the last index)", () => {
  const verifyOnce = async (maxNewTokens: number) => {
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens },
    });
    await assertMatchesPlain({ prompt: PROMPT, maxNewTokens }, run);
    assertEquals(verifyCalls(run.fake).length, 1);
    return { call: verifyCalls(run.fake)[0], run };
  };

  it("k' = 1 (R stays 4, the tail is padded)", async () => {
    const { call, run } = await verifyOnce(3);
    assertEquals(call.queryLength, 2);
    assertEquals(call.lastRows, [0, 1, 1, 1]);
    assertEquals(call.lastRowShape, [K + 1]);
    assertEquals(call.idsShape, [1, 4]);
    assertEquals(call.ids, [6, 8, 0, 0]);
    assertEquals(call.positions, [2, 3, 0, 0]);
    assertEquals(run.fake.commits, [2]);
  });

  it("k' = 2 (only one pad)", async () => {
    const { call, run } = await verifyOnce(4);
    assertEquals(call.queryLength, 3);
    assertEquals(call.lastRows, [0, 1, 2, 2]);
    assertEquals(call.ids, [6, 8, 9, 0]);
    assertEquals(call.positions, [2, 3, 4, 0]);
    assertEquals(run.fake.commits, [3]);
  });

  it("k' = 0 (no pad, the decode-shaped [0])", async () => {
    const { call, run } = await verifyOnce(2);
    assertEquals(call.queryLength, 1);
    assertEquals(call.lastRows, [0]);
    assertEquals(call.lastRowShape, [1]);
    assertEquals(call.idsShape, [1, 1]);
    assertEquals(call.ids, [6]);
    assertEquals(call.positions, [2]);
    assertEquals(run.fake.commits, [1]);
  });
});

// ---- T14: `used` during a run -------------------------------------------------------------

/**
 * Reads `used` **right after** each delivery and checks it is `position + 1` (returns the delivered
 * positions).
 *
 * It reads in the synchronous section after returning from yield, so with speculation it can
 * observe **the window where a pending verify still remains**. If this breaks, a caller that sets
 * the next turn's budget from `used` (the formula in the doc of `GenerationSequence.used`) would
 * let through a limit up to k too lenient in the middle of a cycle.
 */
const drainAssertingUsed = async (
  sequence: GenerationSequence,
  stream: GenerationStream,
): Promise<number[]> => {
  const positions: number[] = [];
  for await (const event of stream) {
    if (event.kind !== "token") continue;
    positions.push(event.position);
    assertEquals(
      sequence.used,
      event.position + 1,
      `used right after delivering the token at position ${event.position}`,
    );
  }
  return positions;
};

describe('T14 used: right after a delivery it is always "that token\'s position + 1"', () => {
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 7 };
  /** The delivered positions (7 of them, right after the 2 prompt tokens). */
  const expected = [2, 3, 4, 5, 6, 7, 8];

  it("speculative (partial acceptance — read in the middle of a cycle)", async () => {
    const opened = await openSpeculative({ drafter: partialDrafter(1) });
    assertEquals(
      await drainAssertingUsed(opened.sequence, opened.sequence.generate(request)),
      expected,
    );
    // The same formula after it settles (one frontier stays uncommitted = it is prepended to the
    // next turn).
    assertEquals(opened.sequence.used, 9);
    assertEquals(opened.fake.pendingCommit(), undefined);
  });

  it("non-speculative (the regression counterpart)", async () => {
    const opened = await openPlain();
    assertEquals(
      await drainAssertingUsed(opened.sequence, opened.sequence.generate(request)),
      expected,
    );
    assertEquals(opened.sequence.used, 9);
  });

  it("after a break in the middle of a cycle it is still the last delivered position + 1", async () => {
    const opened = await openSpeculative({ drafter: oracleDrafter() });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });
    let last = -1;
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      last = event.position;
      assertEquals(opened.sequence.used, last + 1, `right after delivery (position ${last})`);
      // Stop at the second token of cycle 1 (= the window where the pending run still remains).
      if (last === 4) break;
    }
    assertEquals((await stream.done).reason, "closed");
    // A commit moves "uncommitted rows → pastLength", so `used` does not move across the commit.
    assertEquals(opened.fake.commits, [2]);
    assertEquals(
      opened.sequence.used,
      5,
      "used after the break is not the last delivered position + 1",
    );
  });

  it("even when the consumer throws during delivery (the commit in finally), it is the last position + 1", async () => {
    const opened = await openSpeculative({ drafter: oracleDrafter() });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });
    let last = -1;
    await assertRejects(
      async () => {
        for await (const event of stream) {
          if (event.kind !== "token") continue;
          last = event.position;
          // The consumer fails in the same window as the break (the second token of cycle 1 = a
          // pending run remains).
          if (last === 4) throw new Error("consumer");
        }
      },
      Error,
      "consumer",
    );
    // The consumer's exception does not enter the generator (through `return()` = closed). The
    // commit up to the frontier and resetting `frontierRows` to 0 go through the same finally.
    assertEquals((await stream.done).reason, "closed");
    assertEquals(opened.fake.commits, [2]);
    assertEquals(
      opened.sequence.used,
      5,
      "used after the exception is not the last delivered position + 1",
    );
  });
});

// ---- T15: cleanup after a decode whose draw failed ----------------------------------------

/**
 * Folds the valid rows of every run into (position, token) = **the order that entered the KV**
 * (non-speculative runs only — a deferred verify writes rejected rows to the physical ring too, so
 * with speculation it has to be read together with the committed row counts).
 *
 * The same token sitting at two positions is invisible to a comparison of token sequences (the next
 * turn's input is built from `pendingToken`, which never appears in the sequence the consumer
 * received). Folding with positions is the only line that detects it.
 */
const placedRows = (fake: FakeSession): number[][] =>
  fake.calls.flatMap((call) =>
    call.ids.slice(0, call.queryLength).map((id, row) => [call.positions[row], id]),
  );

test("T15 draw failure: when the draw fails after the run went through, the old frontier is not fed again to the next turn", async () => {
  // Fault injection: mix NaN into the logits of the decode run (call 1). The run itself
  // **succeeded**, so the frontier b (= 6) is in the KV at position 2; leaving it in `pendingToken`
  // for the next turn would put the same token at position 3 too (a silent degradation with no
  // exception — the conversation grows by one token).
  const opened = await openPlain({}, { nanAt: 1 });
  await assertRejects(
    () => drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 5 })),
    Error,
    // What fails is the draw (`assertNoNaN`) — not the run (the poisoned id is an implementation
    // detail of the fake, so it is not checked).
    "is NaN (non-finite)",
  );
  assertEquals(
    opened.fake.pastLength(),
    3,
    "the decode run whose draw failed did succeed (the logical length advances)",
  );

  // The next turn on the same sequence (the frontier is already in the KV, so the prompt runs as
  // is).
  const next = await drain(opened.sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  assertEquals(tokenIds(next.events), [SUCCESSOR[4]]);
  assertEquals(placedRows(opened.fake), [
    [0, 1],
    [1, 2],
    [2, 6],
    [3, 4],
  ]);
});

// ---- T16: the self-accounting gate (`policy: "auto"`, fake clock) ------------------------

/**
 * Adds a cycle's cost when a fake target run completes, independent of how often the clock is read.
 */
const fakeClock = (
  drafter: FakeDrafter,
  wallOf: (cycle: number, drafted: boolean) => number,
): { readonly now: () => number; readonly afterRun: () => void } => {
  let elapsed = 0;
  let cycle = 0;
  let draftsAtLastRun = 0;
  return {
    now: (): number => elapsed,
    afterRun: (): void => {
      elapsed += wallOf(cycle, drafter.calls.length > draftsAtLastRun);
      draftsAtLastRun = drafter.calls.length;
      cycle += 1;
    },
  };
};

/**
 * Knobs that make the gate switch without waiting for 2 blocks (the default is 16 cycles × 2 in a
 * row).
 *
 * What is checked here is **the wiring** (accounting of dropped steps, the hidden-state continuity,
 * cycles not mixed into the observation); the decision of "how many cycles to measure before
 * switching" itself belongs to `generation_gate_test.ts`.
 */
const FAST_GATE: SpeculationGateOptions = { window: 2, confirm: 1 };

test("T16 gate: with a wall time where speculation loses it drops to the decode shape, and the sequence still matches the non-speculative one", async () => {
  // A speculative cycle commits 4 in 100ms (25ms/token), a plain step costs 10ms/token = a wall
  // time where speculation is 2.5 times slower.
  const drafter = oracleDrafter();
  const request: GenerationRequest = { prompt: PROMPT, maxNewTokens: 20 };
  const opened = await openSpeculative({
    drafter,
    policy: "auto",
    gate: FAST_GATE,
    clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? 100 : 10)),
  });
  const run = { ...opened, ...(await drain(opened.sequence.generate(request))) };

  // The core of the contract does not change (the gate changes only the speed, not the output).
  await assertMatchesPlain(request, run);
  const always = await runSpeculative({ drafter: oracleDrafter(), request });
  assertEquals(
    tokenIds(run.events),
    tokenIds(always.events),
    "the gated and always sequences differ",
  );

  // The dropped steps are not part of the speculation accounting; `plainSteps` counts them (they do
  // not consume cycle numbers either). The forced plain at the end of the budget (`k' = 0`) counts
  // as a verify as before = `cycles` is 4, and that one goes into `acceptedHistogram[0]`.
  assertEquals(run.stop.speculation, {
    cycles: 4,
    draftRuns: 3,
    drafted: 9,
    accepted: 9,
    delivered: 13,
    acceptedHistogram: [1, 0, 0, 3],
    plainSteps: 6,
    switches: 1,
  });
  // Deliveries are 4 → (the first sample of `W1`) 1 → 4 → 4 (here the second block fills and it
  // switches) → 5 plain → the 1 at the end of the budget (= 20 tokens). The second one is plain
  // because "the first sample of `W1` is the second decision"; the gate switches when it has
  // **observed** the second speculative cycle.
  assertEquals(run.fake.commits, [4, 1, 4, 4, 1, 1, 1, 1, 1, 1]);
  // The gate's plain steps report as `decode` (no branch added to the public type). Their numbers
  // run on from 1.
  assertEquals(
    runShape(run.phases).filter((phase) => phase.kind === "decode"),
    Array.from({ length: 6 }, (_unused, index) => ({ kind: "decode", step: index + 1 })),
  );
  // The verify cycle numbers are 1..4 (plain steps do not skip numbers).
  assertEquals(
    run.phases.filter((phase) => phase.kind === "verify").map((phase) => phase.cycle),
    [1, 2, 3, 4],
  );
});

test("T16 gate: the drafter input (hidden state and frontier) stays connected across plain steps", async () => {
  // The gate's plain step is a one-row verify-shaped run (deferred), so `hidden` can be read.
  // Branching to the non-speculative decode path (which reads only logits) would make the draft of
  // the cycle it returns to eat a stale hidden state. To go as far as "switch → 2 plain steps →
  // explore" within one turn (20 tokens), the burst is shrunk to 1 cycle and the plain-side
  // interval to 2 steps (how many cycles a burst measures is generation_gate_test.ts's concern).
  const drafter = oracleDrafter();
  const opened = await openSpeculative({
    drafter,
    policy: "auto",
    gate: { ...FAST_GATE, burst: 1, exploreBase: 2 },
    clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? 100 : 10)),
  });
  await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 }));

  // The last draft belongs to the exploration burst on the plain side, and right before it come two
  // plain steps (run indices 6 and 7 — index 0 is the prefill). It receives the hidden state of row
  // 0 of run 7 and the frontier run 7 delivered (run 7's input token 3 → `SUCCESSOR[3]` = 13).
  const last = opened.drafter.calls[opened.drafter.calls.length - 1];
  assertEquals(opened.drafter.calls.length, 4);
  assertEquals(last, { token: SUCCESSOR[3], position: 18, hidden: [hiddenMark(7, 0), 3, 0, 0] });
});

describe("T16 gate: the first cycle of a turn and the forced plain at the end of the budget are not mixed into the observation", () => {
  // The first cycle includes the cold miss of the PLE gather, and the forced plain's shape is set
  // by the caller's budget — neither is evidence that "speculation is slow". Checked by fault
  // injection: put 10,000ms on those two kinds only, and the gate never switches (an implementation
  // that mixes them in drops to plain immediately).
  const turn1: GenerationRequest = { prompt: PROMPT, maxNewTokens: 6 };
  const turn2: GenerationRequest = { prompt: [4], maxNewTokens: 20 };
  /**
   * Turn 1 has 2 cycles (the first cycle + the forced plain at the end of the budget); the head of
   * turn 2 is the third.
   */
  const excluded = 3;
  /**
   * The first **speculative cycle that enters the observation** in turn 2 (the first 2 are plain
   * steps that take `W1`).
   */
  const firstMeasuredCycle = 4;

  const play = async (slow: (cycle: number) => boolean) => {
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: fakeClock(drafter, (cycle, drafted) => (slow(cycle) ? 10_000 : drafted ? 20 : 10)),
    });
    await drain(opened.sequence.generate(turn1));
    const second = await drain(opened.sequence.generate(turn2));
    return second.stop.speculation;
  };

  it("not mixed in: turn 2 stays speculative too (it does not switch)", async () => {
    // The first 2 of turn 2 are plain because `W1` is not yet observed (the first is a cycle that
    // is not mixed in, so it returns no observation, and the second finally takes `W1`). The steady
    // re-measurement comes every 16 observations, so it does not fall within a 20-token turn = the
    // only plain steps are the first 2. The last one is the forced plain at the end of the budget
    // (`k' = 0`), counted as a verify and placed in `acceptedHistogram[0]`.
    assertEquals(await play((cycle) => cycle < excluded), {
      cycles: 5,
      draftRuns: 4,
      drafted: 12,
      accepted: 12,
      delivered: 17,
      acceptedHistogram: [1, 0, 0, 4],
      plainSteps: 2,
      switches: 0,
    });
  });

  it("the counterpart (it drops when the same wall time is put on a speculative cycle that enters the observation)", async () => {
    // Putting a single 10,000ms on the first **speculative** cycle that enters the observation in
    // turn 2 makes the gate switch to plain right after that observation. The counterpart showing
    // that the green above is not "a setting that never switches at all". It must not go on a plain
    // step (the cycle that takes `W1`) — with `W1` at 10,000ms the ratio shrinks, speculation looks
    // favourable and it does not switch = no counterpart.
    const speculation = await play((cycle) => cycle === firstMeasuredCycle);
    assertEquals(
      speculation?.switches,
      1,
      "it does not switch even with a slow cycle that enters the observation",
    );
    assert(
      (speculation?.plainSteps ?? 0) > 3,
      `it did not drop to plain after switching: ${JSON.stringify(speculation)}`,
    );
  });

  it("a skipped cycle does not enter the block, but the exploration period advances", async () => {
    // A cycle that is not mixed in still runs one piece of GPU work, so the exploration period
    // advances by it (the gate's `skip()`). Over one turn: the head cycle advances the period by 1
    // without entering the observation, the first sample of `W1` (one plain move) comes **second**,
    // and exploration after that comes every `exploreBase`. An implementation where the period
    // stops shifts both of these back by one.
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      // The block is 2 cycles, so a single 10,000ms at the head mixed in would switch at the first
      // block. The default re-measurement interval (16 observations) does not fit one turn's
      // budget, so the period itself is shrunk to 4 (the default side belongs to T5 of
      // `generation_gate_test.ts`).
      gate: { ...FAST_GATE, exploreBase: 4 },
      clock: fakeClock(drafter, (cycle, drafted) => (cycle === 0 ? 10_000 : drafted ? 20 : 10)),
    });
    const run = {
      ...opened,
      ...(await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 30 }))),
    };
    assertEquals(
      run.phases.flatMap((phase) =>
        phase.kind === "verify" || phase.kind === "decode" ? [phase.kind] : [],
      ),
      [
        "verify",
        "decode",
        "verify",
        "decode",
        "verify",
        "verify",
        "verify",
        "decode",
        "verify",
        "verify",
      ],
      "the exploration period did not advance by the cycles not mixed in",
    );
    assertEquals(
      run.stop.speculation?.switches,
      0,
      "the wall time of a cycle not mixed in entered the block",
    );
  });
});

// ---- T17: wall time and phase on the observation hook (`wallMs` / `delivered` / `gate`) ------

/**
 * Observations of the generation phase (the two kinds that report wall time and phase — prefill and
 * draft do not).
 */
type StepPhase = Extract<GenerationRunPhase, { kind: "decode" | "verify" }>;

const stepsOf = (phases: readonly GenerationRunPhase[]): readonly StepPhase[] =>
  phases.flatMap((phase) => (phase.kind === "decode" || phase.kind === "verify" ? [phase] : []));

/** The gate state of one expected report. */
const gateOf = (
  mode: "speculate" | "plain",
  switches: number,
  measured: boolean,
): GenerationGateTrace => ({ mode, switches, measured });

describe("T17 wall time on the observation hook: the wall times of speculative cycles and plain steps are present and exclude consumer delay", () => {
  it("feeding the same wall times as T16, each run's wallMs matches the fake clock's ticks", async () => {
    // **The same settings** as T16 (speculative cycle 100ms / plain step 10ms, block of 2 cycles).
    // That one checks "the accounting and the decision sequence"; this one checks **the wall time
    // and the phase** of the same run.
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? 100 : 10)),
    });
    const run = {
      ...opened,
      ...(await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 }))),
    };
    const steps = stepsOf(run.phases);

    // Deliveries are 4 → (the first sample of `W1`) 1 → 4 → 4 → 5 plain → the 1 at the end of the
    // budget (T16's commits). The wall time follows that shape: "100ms only for cycles that took a
    // draft".
    assertEquals(
      steps.map((phase) => phase.kind),
      [
        "verify",
        "decode",
        "verify",
        "verify",
        "decode",
        "decode",
        "decode",
        "decode",
        "decode",
        "verify",
      ],
    );
    assertEquals(
      steps.map((phase) => phase.wallMs),
      [100, 10, 100, 100, 10, 10, 10, 10, 10, 10],
      "the run wall times do not match the fake clock's ticks",
    );
  });

  it("the gate state is the value **after** the observation (measured is not set on the first cycle)", async () => {
    const drafter = oracleDrafter();
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? 100 : 10)),
    });
    const run = {
      ...opened,
      ...(await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 }))),
    };

    // At the third speculative cycle (run index 3) the second block fills and it switches = the
    // **result** of that observation is `mode: "plain"` / `switches: 1`. An implementation that
    // reads before the observation shifts it back by one.
    assertEquals(
      stepsOf(run.phases).map((phase) => phase.gate),
      [
        // The first cycle of a turn is not mixed into the wall-time observation (`skip()` — one of
        // the two kinds not mixed in).
        gateOf("speculate", 0, false),
        // The first sample of `W1` (the second decision inserts one plain step).
        gateOf("speculate", 0, true),
        gateOf("speculate", 0, true),
        gateOf("plain", 1, true),
        gateOf("plain", 1, true),
        gateOf("plain", 1, true),
        gateOf("plain", 1, true),
        gateOf("plain", 1, true),
        gateOf("plain", 1, true),
        // The forced plain at the end of the budget is not mixed in either (a cycle whose shape the
        // caller's budget set).
        gateOf("plain", 1, false),
      ],
    );
    assertEquals(
      run.stop.speculation?.switches,
      1,
      "it did not switch at the same position as T16",
    );
  });

  it("the wall time does not move when the consumer advances the clock between deliveries (fault injection)", async () => {
    // An implementation that takes the wall time "after delivery" makes speculation look slower the
    // slower the consumer is (= the gate cuts speculation). The fake clock is advanced in two
    // places, **inside** the cycle (in the draft) and **between** deliveries (the consumer), and
    // only the former must land on the run's wall time. No gate is needed (the wall time is present
    // with `policy: "always"` too).
    const cycleMs = 100;
    const deliveryMs = 1000;
    let elapsed = 0;
    const drafter = fakeDrafter({
      draft: (cycle) => {
        elapsed += cycleMs;
        return chainFrom(cycle.token, K);
      },
    });
    const opened = await openSpeculative({ drafter, now: (): number => elapsed });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 9 });
    let delivered = 0;
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      delivered += 1;
      elapsed += deliveryMs;
    }
    assertEquals((await stream.done).reason, "max-tokens");

    // The fault injection takes effect (the consumer's share is most of the clock).
    assertEquals(delivered, 9);
    assertEquals(elapsed, 2 * cycleMs + 9 * deliveryMs);
    assertEquals(
      stepsOf(opened.phases).map((phase) => phase.wallMs),
      [cycleMs, cycleMs],
      "the run wall times include the delivery (consumer) time",
    );
    // `policy: "always"` creates no gate, so no phase field appears either.
    assertEquals(
      stepsOf(opened.phases).map((phase) => Object.hasOwn(phase, "gate")),
      [false, false],
      "a turn without a gate has a gate field",
    );
  });

  it("non-speculative decode carries the wall time too and has no gate field", async () => {
    const plain = await runPlain({ prompt: PROMPT, maxNewTokens: 3 });
    const steps = stepsOf(plain.phases);
    assertEquals(
      steps.map((phase) => phase.kind),
      ["decode", "decode"],
    );
    for (const phase of steps) {
      const wallMs = phase.wallMs;
      // A fake clock cannot be injected into a non-speculative sequence (`now` is a field of the
      // speculation spec), so only "it is present" is checked, not the actual value. Matching ticks
      // is checked by the speculative cases above.
      assert(
        wallMs !== undefined && Number.isFinite(wallMs) && wallMs >= 0,
        `non-speculative decode carries no wall time: ${JSON.stringify(phase)}`,
      );
      assertEquals(
        Object.hasOwn(phase, "gate"),
        false,
        "a non-speculative observation has a gate field",
      );
    }
  });

  it("delivered is the number of tokens the cycle committed (not derived from accepted + 1)", async () => {
    // A cycle whose enumeration was cut off at a stop token does not commit `accepted + 1` tokens
    // (the same layout as T4 — in the chain 6 → 8 → 9, 9 is the stop token). An implementation that
    // derives it reports one too many here.
    const stopTokens = [9];
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: PROMPT, maxNewTokens: 9 },
      program: { stopTokens },
    });
    const verifies = stepsOf(run.phases).flatMap((phase) =>
      phase.kind === "verify" ? [phase] : [],
    );
    assertEquals(
      verifies.map((phase) => [phase.accepted, phase.delivered]),
      [[2, 2]],
    );
    assertEquals(
      verifies.reduce((sum, phase) => sum + (phase.delivered ?? 0), 0),
      run.stop.speculation?.delivered,
      "the sum of the observed delivered does not match the accounting's delivered",
    );
  });

  it("the sum of delivered matches the accounting (a turn with scattered acceptance)", async () => {
    // A turn of consecutive partially accepted (a = 1) cycles: 2 per cycle = the observed sum lines
    // up with the accounting too.
    const run = await runSpeculative({
      drafter: partialDrafter(1),
      request: { prompt: PROMPT, maxNewTokens: 7 },
    });
    const delivered = stepsOf(run.phases).flatMap((phase) =>
      phase.kind === "verify" ? [phase.delivered] : [],
    );
    assertEquals(delivered, [2, 2, 2]);
    assertEquals(run.stop.speculation?.delivered, 6);
  });
});

// ---- pure functions (`src/generation/speculation.ts`) -------------------------------------

test("planDraftLength: at most k'+1 tokens are committed, so it does not exceed the remaining budget", () => {
  assertEquals(planDraftLength(3, 9), 3, "k itself when the budget has room");
  assertEquals(planDraftLength(3, 4), 3, "exactly k+1 left");
  assertEquals(planDraftLength(3, 3), 2);
  assertEquals(planDraftLength(3, 2), 1);
  assertEquals(
    planDraftLength(3, 1),
    0,
    "with 1 left, no draft is taken and a one-row verify commits it",
  );
  assertEquals(planDraftLength(1, 5), 1);
});

test("verifyRowIndices: R is always k+1 (a cycle where k' shrinks is filled with the last index)", () => {
  assertEquals(verifyRowIndices(3, 3), [0, 1, 2, 3]);
  assertEquals(verifyRowIndices(2, 3), [0, 1, 2, 2]);
  assertEquals(verifyRowIndices(1, 3), [0, 1, 1, 1]);
  // Only `k' = 0` is the decode shape itself (identical to decode down to the PreparedPlan key).
  assertEquals(verifyRowIndices(0, 3), [0]);
  assertEquals(verifyRowIndices(1, 1), [0, 1]);
});

describe("takeDrafts: takes only the first k', and out-of-vocabulary ids or too few drafts fail loudly", () => {
  it("the first k' (the rest is ignored)", () => {
    assertEquals(takeDrafts(Int32Array.from([8, 9, 3]), 2, VOCAB), [8, 9]);
    assertEquals(takeDrafts(Int32Array.from([8, 9, 3]), 0, VOCAB), []);
  });

  it("too few drafts", () => {
    assertThrows(
      () => takeDrafts(Int32Array.from([8, 9]), 3, VOCAB),
      Error,
      "Only 2 drafts (3 needed)",
    );
  });

  it("outside the vocabulary (an out-of-range gather poisons the whole row with NaN)", () => {
    assertThrows(
      () => takeDrafts(Int32Array.from([8, VOCAB]), 2, VOCAB),
      Error,
      `draft[1] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`,
    );
    assertThrows(
      () => takeDrafts(Int32Array.from([-1]), 1, VOCAB),
      Error,
      `draft[0] -1 is outside the vocabulary 0..${VOCAB - 1}`,
    );
  });
});

/**
 * A logits reader where row `j` makes `tokens[j]` the argmax (it also records the indices of the
 * rows read).
 */
const rowsOf = (tokens: readonly number[]) => {
  const read: number[] = [];
  const row = (index: number): Float32Array<ArrayBuffer> => {
    read.push(index);
    if (index >= tokens.length) throw new Error(`read row ${index}, which was not prepared`);
    const data = new Float32Array(new ArrayBuffer(VOCAB * 4));
    data[tokens[index]] = 10;
    return data;
  };
  return { row, read };
};

/**
 * A sampler that records the draw history (actually the real temperature-0 one — the decision is
 * argmax).
 */
const recordingSampler = (): {
  readonly sampler: Pick<Sampler, "next">;
  readonly histories: number[][];
} => {
  const inner = createSampler();
  const histories: number[][] = [];
  return {
    sampler: {
      next: (logits, history): number => {
        histories.push([...history]);
        return inner.next(logits, history);
      },
    },
    histories,
  };
};

/** A turn without any stop token (acceptance is set by the prefix match alone). */
const noStop = (): boolean => false;

describe("acceptDrafts: the position where the prefix match stops is the accepted count, and that row's draw is b′", () => {
  it("all accepted (row k' emits b′)", () => {
    const { row, read } = rowsOf([8, 9, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], noStop), {
      accepted: 3,
      confirmed: [8, 9, 3, 13],
    });
    assertEquals(read, [0, 1, 2, 3], "all accepted reads the draft count + 1 rows");
  });

  it("after a miss midway the later rows are not read (early exit)", () => {
    // Row 1's draw is 7, which disagrees with the draft's 9 → it stops there and b′ = 7.
    const { row, read } = rowsOf([8, 7, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], noStop), {
      accepted: 1,
      confirmed: [8, 7],
    });
    assertEquals(read, [0, 1], "logits after the rejected row were read");
  });

  it("even with 0 accepted, b′ always commits one token (degenerating to one decode step)", () => {
    const { row, read } = rowsOf([8, 9, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(acceptDrafts(row, [REJECTED, REJECTED, REJECTED], sampler, [1, 2, 6], noStop), {
      accepted: 0,
      confirmed: [8],
    });
    assertEquals(read, [0]);
  });

  it('row j\'s draw receives "the committed sequence up to that row"', () => {
    const { row } = rowsOf([8, 9, 3, 13]);
    const { sampler, histories } = recordingSampler();
    const history: readonly number[] = [1, 2, 6];
    acceptDrafts(row, [8, 9, 3], sampler, history, noStop);
    assertEquals(histories, [
      [1, 2, 6],
      [1, 2, 6, 8],
      [1, 2, 6, 8, 9],
      [1, 2, 6, 8, 9, 3],
    ]);
    // The caller's `history` is not touched (the caller extends the committed sequence as it
    // delivers).
    assertEquals(history, [1, 2, 6]);
  });

  it("the draft of a rejected row is not added to the history", () => {
    const { row } = rowsOf([8, 7, 3, 13]);
    const { sampler, histories } = recordingSampler();
    acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], noStop);
    assertEquals(histories, [
      [1, 2, 6],
      [1, 2, 6, 8],
    ]);
  });

  it("when an accepted draft is a stop token the enumeration stops and later rows are not drawn", () => {
    // A reader that does **not prepare** rows 2 and up (reading them throws) = only an
    // implementation that does not stop at the stop turns red. The non-speculative run ends
    // generation at this position, so not touching the logits beyond it is correct.
    const { row, read } = rowsOf([8, 9]);
    const { sampler, histories } = recordingSampler();
    assertEquals(
      acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], (token) => token === 9),
      {
        accepted: 2,
        confirmed: [8, 9],
      },
    );
    assertEquals(read, [0, 1], "rows beyond the stop token were read");
    // Only 2 draws as well (once per committed token = the RNG consumption lines up with the
    // non-speculative run).
    assertEquals(histories, [
      [1, 2, 6],
      [1, 2, 6, 8],
    ]);
  });

  it("when b′ on rejection is a stop token the committed sequence stays accepted + 1 long", () => {
    // It misses at row 1 and draws b′ = 7, and that 7 is a stop token. The sequence is `[d₁, b′]` =
    // length accepted + 1.
    const { row, read } = rowsOf([8, 7, 3, 13]);
    const { sampler } = recordingSampler();
    assertEquals(
      acceptDrafts(row, [8, 9, 3], sampler, [1, 2, 6], (token) => token === 7),
      {
        accepted: 1,
        confirmed: [8, 7],
      },
    );
    assertEquals(read, [0, 1]);
  });
});

test("speculation's diagnostics load does not mix into the cycle wall time or the gate decision", async () => {
  const results = [];
  for (const observerMs of [0, 5, 50]) {
    let elapsed = 0;
    const drafter = fakeDrafter({
      draft: (cycle) => {
        elapsed += 1;
        return chainFrom(cycle.token, K);
      },
    });
    const opened = await openSpeculative({
      drafter,
      policy: "auto",
      gate: FAST_GATE,
      clock: {
        now: () => elapsed,
        afterRun: () => {
          elapsed += 10;
        },
      },
      onRun: () => {
        elapsed += observerMs;
      },
    });
    const stream = opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 60 });
    const events = [];
    for await (const event of stream) events.push(event);
    results.push({ events, stop: await stream.done, phases: opened.phases });
    await opened.sequence.dispose();
  }
  assertEquals(results[1], results[0]);
  assertEquals(results[2], results[0]);
});

test("drawing for an empty draft does not read the history needlessly", () => {
  const history = new Proxy([1, 2, 6], {
    get: () => {
      throw new Error("copied history the draw does not need");
    },
  });
  let draws = 0;
  const { row, read } = rowsOf([8]);
  const result = acceptDrafts(
    row,
    [],
    {
      next: (_logits, seen) => {
        assert(seen === history);
        draws += 1;
        return 8;
      },
    },
    history,
    noStop,
  );
  assertEquals(result, { accepted: 0, confirmed: [8] });
  assertEquals(read, [0]);
  assertEquals(draws, 1);
});

// ---- T18: forking (`fork` — ADR 0083 addendum 2026-10-11, ruling 8) ---------------------------
//
// A fork's drafter is **reopened on the duplicated context** (borrowers are per context). A drafter
// mix-up does not show in the token sequence (drafts affect only the acceptance rate), so it can
// only be pinned through the context `open` received and "which drafter drafted". The gate copies
// its observation state (it continues from where its decisions left off).

test("T18 fork: the drafter is reopened on the duplicated context and the original drafter is not shared", async () => {
  const fake = fakeSession({ successor: SUCCESSOR });
  const drafters = [oracleDrafter(), oracleDrafter()];
  for (const drafter of drafters) drafter.watch(fake);
  const seen: unknown[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake),
    speculative: {
      open: (context) => {
        seen.push(context);
        const drafter = drafters[seen.length - 1];
        assert(drafter !== undefined, "open was called more often than expected");
        return Promise.resolve(drafter.face);
      },
      policy: "always",
    },
  });
  await drain(sequence.generate({ prompt: PROMPT, maxNewTokens: 6 }));
  const forked = await sequence.fork();
  assertEquals(seen.length, 2, "the fork did not reopen the drafter");
  assert(
    seen[0] === fake.context,
    "the original drafter opened on a context other than the original",
  );
  assert(
    seen[1] === fake.forks[0],
    "the fork's drafter opened on a context other than the duplicate",
  );

  // The drafts of the fork's turn come only from the fork's drafter, and the verify runs on the
  // duplicated context.
  const draftsOfMain = drafters[0].calls.length;
  const atBranch = fake.calls.length;
  const branch = await drain(forked.generate({ prompt: [3], maxNewTokens: 6 }));
  assertEquals(
    drafters[0].calls.length,
    draftsOfMain,
    "the original drafter drafted on the fork's turn",
  );
  assert(drafters[1].calls.length > 0, "the fork's drafter never drafted");
  assert(
    fake.runContexts.slice(atBranch).every((context) => context === fake.forks[0]),
    "the fork's turn ran on a context other than the duplicate",
  );
  // With the same state and the same request, it gives the same sequence as the original
  // (speculation does not change the sequence — on a fork either).
  const main = await drain(sequence.generate({ prompt: [3], maxNewTokens: 6 }));
  assertEquals(
    branch.events,
    main.events,
    "the fork and the original give different sequences for the same request",
  );

  // Lifetime: the fork's dispose tears down only the fork's drafter → the duplicated context.
  await forked.dispose();
  assertEquals([drafters[0].disposals(), drafters[1].disposals()], [0, 1]);
  assertEquals([fake.disposals(), fake.disposalsOf(fake.forks[0])], [0, 1]);
  await sequence.dispose();
  assertEquals(drafters[0].disposals(), 1);
});

test("T18 fork: if reopening the drafter fails, the duplicated context is torn down before throwing and the original is untouched", async () => {
  const fake = fakeSession({ successor: SUCCESSOR });
  const drafter = oracleDrafter();
  drafter.watch(fake);
  let opens = 0;
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: specProgram(fake),
    speculative: {
      open: () => {
        opens += 1;
        return opens === 1
          ? Promise.resolve(drafter.face)
          : Promise.reject(new Error("injected: reopening failed"));
      },
      policy: "always",
    },
  });
  await drain(sequence.generate({ prompt: PROMPT, maxNewTokens: 4 }));
  const used = sequence.used;

  await assertRejects(() => sequence.fork(), Error, "injected: reopening failed");
  assertEquals(fake.forks.length, 1);
  assertEquals(
    fake.disposalsOf(fake.forks[0]),
    1,
    "the context of the fork that could not be opened leaked",
  );
  assertEquals(drafter.disposals(), 0, "the original drafter was torn down as collateral");
  assertEquals(fake.disposals(), 0, "the original context was torn down as collateral");
  // The original can continue as is.
  assertEquals(sequence.used, used);
  const next = await drain(sequence.generate({ prompt: [3], maxNewTokens: 2 }));
  assertEquals(next.stop.reason, "max-tokens");
  await sequence.dispose();
});

test("T18 fork: the gate's observation state is inherited (the fork's first cycle is set by the original gate's mode)", async () => {
  // Switch the original gate to plain with the same wall times as T16 (speculation 2.5 times
  // slower), then fork.
  const drafter = oracleDrafter();
  const opened = await openSpeculative({
    drafter,
    policy: "auto",
    gate: FAST_GATE,
    clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? 100 : 10)),
  });
  const first = await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 }));
  assertEquals(
    first.stop.speculation?.switches,
    1,
    "premise: the original gate did not switch to plain",
  );

  const forked = await opened.sequence.fork();
  const mark = opened.phases.length;
  const branch = await drain(forked.generate({ prompt: [3], maxNewTokens: 4 }));
  /**
   * Reports that name a cycle (draft is an auxiliary report that arrives before the verify, so it
   * is excluded).
   */
  const isCycle = (phase: GenerationRunPhase): boolean =>
    phase.kind === "verify" || phase.kind === "decode";
  const cycles = opened.phases.slice(mark).filter(isCycle);
  // The fork's first cycle is set by the original gate's steady mode (plain) = the decode shape. An
  // implementation that re-measures takes a new gate's first move (speculate = verify).
  const head = cycles[0];
  assert(
    head?.kind === "decode",
    `the fork's first cycle is not the decode shape: ${JSON.stringify(head)}`,
  );
  assertEquals(head.gate?.mode, "plain");
  assertEquals(branch.stop.speculation?.switches, 0, "a switch happened on the fork's turn");

  // Positive control: the first cycle of a gate without observations (a new sequence) is
  // speculative (verify).
  const fresh = await openSpeculative({
    drafter: oracleDrafter(),
    policy: "auto",
    gate: FAST_GATE,
  });
  await drain(fresh.sequence.generate({ prompt: PROMPT, maxNewTokens: 4 }));
  assertEquals(
    fresh.phases.filter(isCycle)[0]?.kind,
    "verify",
    "the first move of a gate without observations is not speculative (the check above would miss)",
  );
  await forked.dispose();
  await fresh.sequence.dispose();
  await opened.sequence.dispose();
});

test("T18 fork: the gate is copied, not shared (switching only the fork does not move the original's mode)", async () => {
  // With wall times where speculation wins (a speculative cycle commits 4 in 20ms, plain step
  // 10ms/token) the original stays speculative, and only the fork's turn gets T16's wall times
  // (speculation 100ms) so the fork's gate switches to plain. In an implementation that shares the
  // gate instead of copying it at the sequence layer (a regression from `gate?.fork()` to `gate`),
  // what switched is the original gate itself, so the first cycle of the original's next turn
  // becomes the decode shape (plain).
  // NOTE: the report's `gate.switches` and the stop's `switches` are counts **within the turn**
  // (the difference from the running total at the start of the turn), so they cannot show that the
  // fork's switch entered the original's total — what is pinned is the mode, and that no switch
  // happens within the original's turn.
  const drafter = oracleDrafter();
  let slow = false;
  const opened = await openSpeculative({
    drafter,
    policy: "auto",
    gate: FAST_GATE,
    clock: fakeClock(drafter, (_cycle, drafted) => (drafted ? (slow ? 100 : 20) : 10)),
  });
  const first = await drain(opened.sequence.generate({ prompt: PROMPT, maxNewTokens: 20 }));
  assertEquals(
    first.stop.speculation?.switches,
    0,
    "premise: the original gate did not stay speculative",
  );
  assertEquals(
    stepsOf(opened.phases).at(-1)?.gate?.mode,
    "speculate",
    "premise: the original gate's steady mode is not speculative",
  );

  const forked = await opened.sequence.fork();
  slow = true;
  const branch = await drain(forked.generate({ prompt: [3], maxNewTokens: 20 }));
  assertEquals(
    branch.stop.speculation?.switches,
    1,
    "premise: the fork's gate did not switch to plain",
  );

  slow = false;
  const mark = opened.phases.length;
  const next = await drain(opened.sequence.generate({ prompt: [3], maxNewTokens: 4 }));
  const head = stepsOf(opened.phases.slice(mark))[0];
  assert(
    head?.kind === "verify",
    `the first cycle of the original's next turn is not speculative: ${JSON.stringify(head)}`,
  );
  assertEquals(head.gate?.mode, "speculate");
  assertEquals(next.stop.speculation?.switches, 0, "a switch happened on the original's turn");
  await forked.dispose();
  await opened.sequence.dispose();
});

// ---- a request without maxNewTokens (ADR 0083 addendum 2026-11-05) ----------------------

describe("speculation without maxNewTokens runs to the limit", () => {
  /** PROMPT is 2 tokens, so the limit is K = 16 − 2 + 1 = 15. */
  const CAPACITY = 16;
  const LIMIT = 15;
  const program = { capacity: CAPACITY };

  for (const [name, drafter] of [
    ["all accepted", (): FakeDrafter => oracleDrafter()],
    ["none accepted", (): FakeDrafter => wrongDrafter()],
    ["partly accepted", (): FakeDrafter => partialDrafter(1)],
  ] as const) {
    it(`yields the tokens of an explicit limit and asks for no row past the capacity (${name})`, async () => {
      const absent = await runSpeculative({
        drafter: drafter(),
        request: { prompt: PROMPT },
        program,
      });
      const explicit = await runSpeculative({
        drafter: drafter(),
        request: { prompt: PROMPT, maxNewTokens: LIMIT },
        program,
      });

      assertEquals(absent.stop.reason, "capacity");
      assertEquals(absent.stop.tokens, LIMIT);
      assertEquals(explicit.stop.reason, "max-tokens");
      assertEquals(tokenIds(absent.events), tokenIds(explicit.events));
      assertEquals(absent.fake.calls, explicit.fake.calls);
      assertEquals(absent.fake.commits, explicit.fake.commits);
      assert(
        absent.fake.calls.every((call) => call.pastBefore + call.queryLength <= CAPACITY),
        "a run reached past the capacity",
      );
      assertEquals(absent.fake.pastLength(), CAPACITY);
      assertEquals(absent.fake.pendingCommit(), undefined);
    });
  }

  it("a prompt with K = 1 takes no draft", async () => {
    const run = await runSpeculative({
      drafter: oracleDrafter(),
      request: { prompt: Array.from({ length: CAPACITY }, (_unused, index) => (index % 14) + 1) },
      program,
    });
    assertEquals(run.stop.reason, "capacity");
    assertEquals(run.stop.tokens, 1);
    assertEquals(run.drafter.calls.length, 0);
  });
});
