/**
 * The fakes (static wiring, Session, context) shared by the GPU-free tests of the generation
 * surface (`src/generation/sequence.ts`). This is where the non-speculative contract
 * (`generation_sequence_test.ts`) and the speculative-path contract
 * (`generation_speculation_test.ts`) point at **the same implementation**.
 *
 * The generation loop takes narrow interfaces (`GenerationSession` / `GenerationContextFace`), so a
 * fake needs only one plain object. This file holds four things:
 *
 * - **Advancing the logical length**: a successful `run` advances `pastLength`. A
 *   `commit: "deferred"` run does not advance it but sets a pending commit, and `commit(rows)`
 *   advances it by the accepted row count (the same order as the real `GenerationContext`). A run
 *   while a commit is pending, a `commit` with nothing pending, and a deferred run beyond the
 *   sliding ring's slack fail loudly like the real context / real executor (down to the checks'
 *   wording). The logical length and the pending commit are kept **per context** (`fork` returns an
 *   independent context with the logical length copied — ADR 0066 addendum 2026-10-11;
 *   `createGenerationContext` always returns the first one, {@link FakeSession.context}). `run`
 *   accepts **only contexts this fake made** (the first one and the branches from `fork` — others
 *   fail loudly). The fake `fork` copies **only two rejections: while a commit is pending and after
 *   dispose**; it does not copy the rest of the real `GenerationContext.fork`'s rejections
 *   (poisoning, an unsettled run, a batch reservation, borrowers, device loss, a disposed Session)
 *   — those belong to the real-GPU checks (`packages/runtime/tests/gpu_generation_context_test.ts`,
 *   and for borrowers `packages/runtime/tests/gpu_borrowed_context_test.ts`).
 * - **Per-row logits**: passing {@link FakeOptions.successor} makes it a map "the row's input token
 *   → the row's argmax" (= a deterministic target model decided by the previous token alone).
 *   The rows of a verify can be expressed by one successor function, so speculative acceptance and
 *   rejection can be set up with one array.
 * - **Per-row hidden**: writes a different value per row ({@link hiddenMark}). "Which row's hidden
 *   the generation loop copied" can be observed only here.
 * - **Recording position sequences**: positions are not a run input but go to the derived-input
 *   hook (`derivedInputs`), so this fake is the only place where "which positions went to which
 *   run" can be seen.
 */

import type { GenerationContextSpec, RunInputs, RunOutputs, SymbolBindings } from "@karume/runtime";
import {
  createGenerationProgram,
  type DerivedRunInputs,
  type GenerationGraph,
  type GenerationProgramSpec,
  type GenerationWiring,
} from "../../src/generation/program.ts";
import type {
  GenerationEvent,
  GenerationGreedyAhead,
  GenerationRunPhase,
  GenerationSession,
  GenerationStop,
  GenerationStream,
  GreedyRunResult,
} from "../../src/generation/sequence.ts";

export const VOCAB = 16;
export const HIDDEN_SIZE = 4;
export const IDS = "input_ids";
export const LAST_ROW = "last_row";
export const LOGITS = "logits";
export const HIDDEN = "hidden";
export const DERIVED = "per_layer_inputs";
export const CHUNK_LENGTH = 4;

/**
 * The default slack of the sliding ring (a value large enough not to hit the check — only the
 * check's tests make it small).
 */
export const SLIDING_SLACK = 64;

export const graphOf = (): GenerationGraph => ({
  symbols: ["C", "M", "R"],
  inputs: [
    { name: IDS, dtype: "i32", shape: [1, "M"] },
    { name: DERIVED, dtype: "f32", shape: [1, "M", 2] },
    // The row selection is an index **sequence** (`R` = the number of rows the run selects). Always
    // 1 for non-speculative, k+1 for verify.
    { name: LAST_ROW, dtype: "i32", shape: ["R"] },
  ],
  outputs: [LOGITS, HIDDEN],
  values: {
    [LOGITS]: { dtype: "f32", shape: [1, "R", VOCAB] },
    [HIDDEN]: { dtype: "f32", shape: [1, "R", HIDDEN_SIZE] },
  },
});

/**
 * The static wiring. **The fake owns the derived-input hook** (by default an implementation that
 * records the position sequences).
 *
 * Positions are no longer a run input (`position_ids` left the graph, and the host inputs that
 * depend on positions are taken by the derived-input hook), so this is the only place where "which
 * positions went to which run" can be seen.
 */
export const programOf = (
  fake: FakeSession,
  override: Partial<GenerationProgramSpec> = {},
): GenerationWiring =>
  createGenerationProgram({
    graph: graphOf(),
    inputIds: IDS,
    lastRow: LAST_ROW,
    logits: LOGITS,
    hidden: HIDDEN,
    chunkLength: CHUNK_LENGTH,
    maxPosition: 128,
    capacity: 64,
    vocabSize: VOCAB,
    stopTokens: [],
    capacitySymbol: "C",
    derivedInputs: fake.derivedInputs,
    ...override,
  });

/**
 * The record of one run (fine-grained enough that the call list alone reveals the whole shape of a
 * step).
 */
export type RunCall = {
  readonly ids: readonly number[];
  readonly idsShape: readonly number[];
  readonly positions: readonly number[];
  readonly lastRow: number;
  /** **All** the indices passed to `last_row` (k+1 for verify — the tail may be padded). */
  readonly lastRows: readonly number[];
  readonly lastRowShape: readonly number[];
  readonly queryLength: number;
  /**
   * `"deferred"` = a run that holds the logical length pending (the speculative verify);
   * `undefined` = a run that advances it on the spot.
   */
  readonly commit: "deferred" | undefined;
  readonly pastBefore: number;
  readonly bindings: SymbolBindings | undefined;
  readonly extra: readonly string[];
  readonly sameContext: boolean;
};

/**
 * A deferred run's pending commit (the same shape as the real `GenerationContext.pendingCommit`).
 */
type FakePending = { readonly pastLength: number; readonly queryLength: number };

/**
 * The fake's context (`GenerationContextFace` + `fork`). The logical length and the pending commit
 * are kept per context.
 *
 * The type is spelled out because `fork` returns its own type (F-bound) — left to inference, the
 * recursive type cannot be written.
 */
export type FakeContext = {
  readonly pastLength: number;
  readonly pendingCommit: FakePending | undefined;
  readonly slidingSlack: number | undefined;
  commit(rows: number): void;
  /**
   * Returns an independent context with the logical length copied (like the real
   * `GenerationContext.fork`, it rejects while a commit is pending and after dispose). These are
   * the only two rejections copied (the rest: "Advancing the logical length" at the top of the
   * file).
   */
  fork(): Promise<FakeContext>;
  dispose(): Promise<void>;
};

const readRow = (
  inputs: RunInputs,
  name: string,
): { readonly shape: readonly number[]; readonly values: readonly number[] } => {
  if (!Object.hasOwn(inputs, name)) throw new Error(`fake: input '${name}' was not passed`);
  const tensor = inputs[name];
  if (!("data" in tensor)) throw new Error(`fake: input '${name}' is not a host tensor`);
  if (tensor.dtype !== "i32") throw new Error(`fake: input '${name}' is not i32`);
  return { shape: tensor.shape, values: [...tensor.data] };
};

/**
 * The per-row mark of hidden (decided by the `run`'s number and **the input row's index** — the
 * detection line for mixed-up rows).
 *
 * The generation loop reads "output row n", but its contents are those of the input row that
 * `last_row[n]` pointed at. Making the mark from the input row's index turns an implementation that
 * broke the output-row/input-row correspondence directly into a difference in values.
 */
export const hiddenMark = (call: number, inputRow: number): number => call * 100 + inputRow + 1;

export type FakeOptions = {
  /**
   * The token id each run's argmax should point at (looked up by call number —
   * {@link FakeOptions.successor} takes precedence).
   */
  readonly tokens?: readonly number[];
  /**
   * A map "the row's input token → the row's argmax" (length `VOCAB`).
   *
   * A deterministic target model where the next token is decided by the previous token alone. Row
   * `j` of a verify is the row that consumes token `[b, d₁..][j]`, so this one map expresses
   * "accepted when the draft hits, rejected when it misses", and the fake side does not itself
   * guarantee that speculative and non-speculative produce **the same token sequence** (an
   * implementation that mixes up rows breaks the sequence).
   */
  readonly successor?: readonly number[];
  /** Makes the run with this count (0-based) fail. */
  readonly failAt?: number;
  /**
   * Mixes NaN into the logits of the run with this count (0-based) (**the run succeeds and sampling
   * fails**).
   *
   * The difference from {@link FakeOptions.failAt} is the point of this option — the run went
   * through, so the logical length advances, and what fails is `sampler.next` (`assertNoNaN`). This
   * is the shape that per-row NaN poisoning from an out-of-range gather actually takes, and it is
   * the only detection line for the clean-up when "the run succeeded but no token could be chosen".
   */
  readonly nanAt?: number;
  /**
   * Poisons only this **output row** of the {@link FakeOptions.nanAt} run (every row of that run
   * when omitted).
   *
   * A speculative verify has k+1 rows in one run, so being able to poison only "a row the
   * non-speculative path would not touch" makes "whether `sampler.next` was called on that row"
   * readable from whether an exception occurs (only an implementation that does not call it goes
   * green).
   */
  readonly nanRow?: number;
  /**
   * The runner-up (an id with a lower logit than the first candidate).
   *
   * The default logits are "10 for the targeted id only, 0 for all others", so a repetition penalty
   * that divides positive values does not change the ranking (10/penalty > 0), and with temperature
   * 0 argmax its effect is **in principle unobservable**. Placing a second candidate makes it
   * possible to see deterministically whether the penalty pushed the first candidate below it.
   */
  readonly runnerUp?: { readonly id: number; readonly logit: number };
  /**
   * Returns logits / hidden with **this many rows** (by default the number of `last_row` indices
   * passed = the correct implementation).
   *
   * An option for fault injection — the form "asked for 1 row, got R rows" is valid as a graph
   * declaration (R is a symbol), so only the row-count check of `readLogits` is a detection line.
   */
  readonly logitsRows?: number;
  /**
   * The number of the run {@link FakeOptions.logitsRows} applies to (0-based; every run when
   * omitted).
   */
  readonly logitsRowsAt?: number;
  /**
   * The sliding ring's slack ({@link SLIDING_SLACK} when omitted).
   *
   * **Passing `undefined` explicitly** gives a context without a sliding slot (presence of the
   * field is what distinguishes them, so omitting it and passing it explicitly are different).
   */
  readonly slidingSlack?: number | undefined;
  /**
   * Where the dispose order is recorded (the first context pushes `"context"`, contexts made by
   * `fork` push `"context:fork1"`, `"context:fork2"`… in the order they were made).
   */
  readonly disposeLog?: string[];
};

export type FakeSession = ReturnType<typeof fakeSession>;

/** One context's mutable state (logical length, pending commit, dispose count). */
type FakeContextState = {
  pastLength: number;
  pending: FakePending | undefined;
  disposals: number;
};

export const fakeSession = (options: FakeOptions = {}) => {
  const calls: RunCall[] = [];
  const specs: GenerationContextSpec[] = [];
  /**
   * The accepted row counts passed to `commit(rows)` (one per cycle — the observation point for the
   * speculative checks).
   */
  const commits: number[] = [];
  /**
   * The state of the contexts this fake made (a run moves only the state of the context it was
   * given).
   */
  const states = new Map<FakeContext, FakeContextState>();
  /** The contexts made by `fork` (in the order they were made). */
  const forks: FakeContext[] = [];
  /**
   * The context each run ran on (same indices as `calls`). It is not put in `RunCall` so that the
   * checks that compare whole call lists between fakes do not break on the identity of the
   * contexts.
   */
  const runContexts: FakeContext[] = [];
  const stateOf = (context: FakeContext): FakeContextState => {
    const state = states.get(context);
    if (state === undefined) throw new Error("fake: a context this fake did not make");
    return state;
  };
  /**
   * The positions the last `derive` received (merged into the run's record — positions are not a
   * run input).
   */
  let derivedPositions: readonly number[] = [];
  /**
   * The default derived-input hook (returns `[1,M,2]` f32 while recording the positions passed).
   */
  const derivedInputs: DerivedRunInputs = {
    names: [DERIVED],
    derive: (ids, positions) => {
      if (ids.length !== positions.length) {
        throw new Error(
          `fake: the lengths of ids ${ids.length} and positions ${positions.length} differ`,
        );
      }
      derivedPositions = [...positions];
      // Column 0 of each row carries its position: a run-ahead step is derived before it runs, so
      // the fake reads the position from the step's own input instead of the latest derive.
      const data = new Float32Array(ids.length * 2);
      positions.forEach((position, row) => {
        data[row * 2] = position;
      });
      return Promise.resolve({
        [DERIVED]: { dtype: "f32", shape: [1, ids.length, 2], data },
      } satisfies RunInputs);
    },
  };
  // Undeclared: a value large enough (does not hit the check). An explicit `undefined` means no
  // sliding slot.
  const slidingSlack = Object.hasOwn(options, "slidingSlack")
    ? options.slidingSlack
    : SLIDING_SLACK;
  /**
   * Builds one context (state is read and written only through `state` — branches do not share
   * state).
   */
  const makeContext = (state: FakeContextState, label: string): FakeContext => {
    const made: FakeContext = {
      get pastLength(): number {
        return state.pastLength;
      },
      get pendingCommit(): FakePending | undefined {
        return state.pending;
      },
      slidingSlack,
      /**
       * Commits the first `rows` rows of the pending deferred run (the same accepted set as the
       * real `GenerationContext.commit` — a call with nothing pending or out of range fails
       * loudly).
       */
      commit: (rows: number): void => {
        const pending = state.pending;
        if (pending === undefined) {
          throw new Error(
            "fake: no generation run awaiting commit (only a run issued as deferred needs a commit)",
          );
        }
        if (!Number.isSafeInteger(rows) || rows < 0 || rows > pending.queryLength) {
          throw new Error(`fake: accepted row count ${rows} is outside 0..${pending.queryLength}`);
        }
        state.pastLength = pending.pastLength + rows;
        state.pending = undefined;
        commits.push(rows);
      },
      fork: (): Promise<FakeContext> => {
        // The same rejections as the real `GenerationContext.fork` (decided at the call, returned
        // through the Promise).
        if (state.disposals > 0) {
          return Promise.reject(new Error("fake: a disposed context cannot be forked"));
        }
        if (state.pending !== undefined) {
          return Promise.reject(
            new Error("fake: cannot fork while a generation run awaits commit"),
          );
        }
        const forked = makeContext(
          { pastLength: state.pastLength, pending: undefined, disposals: 0 },
          `context:fork${forks.length + 1}`,
        );
        forks.push(forked);
        return Promise.resolve(forked);
      },
      dispose: (): Promise<void> => {
        state.disposals += 1;
        options.disposeLog?.push(label);
        return Promise.resolve();
      },
    };
    states.set(made, state);
    return made;
  };
  const rootState: FakeContextState = { pastLength: 0, pending: undefined, disposals: 0 };
  const context = makeContext(rootState, "context");
  /**
   * One run (the body of `session.run`, and of each step of {@link fakeSession}'s `greedyAhead`).
   * `chain` is the runtime's `GenerationRun.chain`: a deferred run placed after the pending rows,
   * extending them (only the run-ahead passes it — the sequence's own runs never chain).
   */
  const runStep = (
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation: {
      readonly context: FakeContext;
      readonly queryLength: number;
      readonly commit?: "deferred";
      readonly chain?: boolean;
    },
  ): RunOutputs => {
    // The logical length and the pending commit are **the given context's** (branched contexts do
    // not share state).
    const state = stateOf(generation.context);
    const pending = state.pending;
    // MUST: the same rejection as the real context (a second run while a commit is pending would
    // capture the logical length without knowing how much is committed). Not recorded either = the
    // same treatment as a run that fails in the synchronous part of issuing.
    if (pending !== undefined && generation.chain !== true) {
      throw new Error(`fake: a generation run awaits commit (queryLength ${pending.queryLength})`);
    }
    // MUST: a deferred run's row count is bounded by the sliding ring's slack (copies the real
    // runtime's check **with the same wording** — the deferred section of `executor.ts`). Rejected
    // rows overwrite live past KV in the physical ring, so with a fake that lacks this, a
    // regression that drops `k + 1 ≤ slidingSlack` stays green without a GPU and first fails at the
    // first verify on a real GPU (at the end of a GB-scale load).
    if (generation.commit === "deferred") {
      const slack = generation.context.slidingSlack;
      if (slack !== undefined && generation.queryLength > slack) {
        throw new Error(
          `run: queryLength ${generation.queryLength} of a deferred generation run exceeds ` +
            `the sliding ring's slack ${slack} (rejected rows would overwrite live past KV)`,
        );
      }
      // A chain is bounded as a whole (the runtime's `acquireChainedRun`, with its wording).
      const rows = (pending?.queryLength ?? 0) + generation.queryLength;
      if (generation.chain === true && slack !== undefined && rows > slack) {
        throw new Error(
          `run: the chained run would leave ${rows} rows uncommitted, more than the sliding ring's slack ${slack}`,
        );
      }
    }
    // A chained run sits after the pending rows (the position the runtime reserves for it).
    const position =
      pending === undefined ? state.pastLength : pending.pastLength + pending.queryLength;
    const call = calls.length;
    const ids = readRow(inputs, IDS);
    const lastRow = readRow(inputs, LAST_ROW);
    calls.push({
      ids: ids.values,
      idsShape: ids.shape,
      // The positions `derive` received right before this run (not among the run's inputs).
      positions: derivedPositions,
      lastRow: lastRow.values[0],
      lastRows: lastRow.values,
      lastRowShape: lastRow.shape,
      queryLength: generation.queryLength,
      commit: generation.commit,
      pastBefore: position,
      bindings,
      extra: Object.keys(inputs).filter((name) => name !== IDS && name !== LAST_ROW),
      sameContext: generation.context === context,
    });
    runContexts.push(generation.context);
    if (options.failAt === call) throw new Error("run が落ちた");
    // The logical length advances when the run succeeds (deferred holds it pending and waits for
    // commit — like the real context).
    // A chained run extends the pending rows (only a chained run reaches here with some pending).
    if (generation.commit === "deferred") {
      state.pending =
        pending === undefined
          ? { pastLength: state.pastLength, queryLength: generation.queryLength }
          : {
              pastLength: pending.pastLength,
              queryLength: pending.queryLength + generation.queryLength,
            };
    } else {
      state.pastLength += generation.queryLength;
    }
    // The number of rows returned is the number of `last_row` indices passed (same as the real
    // graph = R is bound by the input).
    const rows =
      options.logitsRows !== undefined &&
      (options.logitsRowsAt === undefined || options.logitsRowsAt === call)
        ? options.logitsRows
        : lastRow.values.length;
    const logits = new Float32Array(rows * VOCAB);
    const hidden = new Float32Array(rows * HIDDEN_SIZE);
    for (let row = 0; row < rows; row += 1) {
      // Output row n carries the contents of the **input row** `last_row[n]` points at (under the
      // fault injection that fakes the row count the indices run out, and those rows fold to input
      // row 0).
      const inputRow = lastRow.values[row] ?? 0;
      const id =
        options.successor === undefined
          ? (options.tokens?.[call] ?? (call + 1) % VOCAB)
          : options.successor[ids.values[inputRow]];
      logits[row * VOCAB + id] = 10;
      if (options.runnerUp !== undefined) {
        logits[row * VOCAB + options.runnerUp.id] = options.runnerUp.logit;
      }
      // The poison goes on **another** id of the row (poisoning the targeted id would break the
      // "row argmax" map).
      if (options.nanAt === call && (options.nanRow === undefined || options.nanRow === row)) {
        logits[row * VOCAB + ((id + 1) % VOCAB)] = Number.NaN;
      }
      hidden[row * HIDDEN_SIZE] = hiddenMark(call, inputRow);
      hidden[row * HIDDEN_SIZE + 1] = ids.values[inputRow];
    }
    return {
      [LOGITS]: { dtype: "f32", shape: [1, rows, VOCAB], data: logits },
      // Only the speculative path reads hidden (an implementation that regresses to reading it on
      // the non-speculative path hits the logits shape check with this per-row value).
      [HIDDEN]: { dtype: "f32", shape: [1, rows, HIDDEN_SIZE], data: hidden },
    };
  };
  const session: GenerationSession<FakeContext> = {
    createGenerationContext: (spec) => {
      specs.push(spec);
      return Promise.resolve(context);
    },
    // deno-lint-ignore require-await
    run: async (inputs, bindings, generation): Promise<RunOutputs> =>
      runStep(inputs, bindings, generation),
  };
  /** The step count of every `greedyAhead` call (one entry per batch). */
  const aheadBatches: number[] = [];
  /**
   * The decode run-ahead on this fake: each step is a chained deferred run fed the previous step's
   * {@link fakeArgmax} (the on-GPU selection), so the batch leaves its rows pending exactly as the
   * runtime does. A step that fails rolls the batch's rows back (a failed batch folds none).
   */
  const greedyAhead = (depth: number): GenerationGreedyAhead<FakeContext> => ({
    depth,
    // deno-lint-ignore require-await
    run: async ({ token, steps, context: target }) => {
      aheadBatches.push(steps.length);
      const state = stateOf(target);
      const before = state.pending;
      const selections: GreedyRunResult[] = [];
      let fed = token;
      try {
        for (const step of steps) {
          if (Object.hasOwn(step, IDS)) {
            throw new Error("fake: a run-ahead step carries token ids (its token is on the GPU)");
          }
          derivedPositions = positionsOf(step);
          const outputs = runStep(
            { ...step, [IDS]: { dtype: "i32", shape: [1, 1], data: Int32Array.of(fed) } },
            undefined,
            { context: target, queryLength: 1, commit: "deferred", chain: true },
          );
          const selected = fakeArgmax(outputs);
          selections.push(selected);
          fed = selected.index;
        }
      } catch (error) {
        state.pending = before;
        throw error;
      }
      return selections;
    },
  });
  return {
    session,
    greedyAhead,
    aheadBatches,
    derivedInputs,
    /**
     * The only instance `createGenerationContext` returns (it must be the same surface the
     * borrower's `open` receives). The `disposals` / `pastLength` / `pendingCommit` below are this
     * context's values.
     */
    context,
    /** The contexts made by `fork` (in the order they were made). */
    forks,
    calls,
    runContexts,
    specs,
    commits,
    disposals: (): number => rootState.disposals,
    pastLength: (): number => rootState.pastLength,
    pendingCommit: (): FakePending | undefined => rootState.pending,
    /** The dispose count of any context (including branches). */
    disposalsOf: (target: FakeContext): number => stateOf(target).disposals,
  };
};

/** The positions a step's derived input carries (column 0 of each row — the default derive). */
const positionsOf = (inputs: RunInputs): number[] => {
  const tensor = Object.hasOwn(inputs, DERIVED) ? inputs[DERIVED] : undefined;
  if (tensor === undefined || !("data" in tensor) || tensor.dtype !== "f32") {
    throw new Error(`fake: a run-ahead step has no f32 '${DERIVED}' to read its position from`);
  }
  return Array.from({ length: tensor.data.length / 2 }, (_, row) => tensor.data[row * 2]);
};

/**
 * The selection of one logits row as the GPU selector reports it: the first maximum, or the first
 * NaN with its value (so the host's draw sees the NaN and throws, as the per-step greedy does).
 */
export const fakeArgmax = (outputs: RunOutputs): GreedyRunResult => {
  const logits = outputs[LOGITS];
  if (logits.dtype !== "f32") throw new Error("fake: logits are not f32");
  let index = 0;
  let value = -Infinity;
  for (let at = 0; at < logits.data.length; at += 1) {
    if (Number.isNaN(logits.data[at])) return { index: at, value: logits.data[at] };
    if (logits.data[at] > value) {
      index = at;
      value = logits.data[at];
    }
  }
  return { index, value };
};

/** Drains every event (returns `done` too). */
export const drain = async (
  stream: GenerationStream,
): Promise<{ readonly events: GenerationEvent[]; readonly stop: GenerationStop }> => {
  const events: GenerationEvent[] = [];
  for await (const event of stream) events.push(event);
  return { events, stop: await stream.done };
};

export const tokenIds = (events: readonly GenerationEvent[]): number[] =>
  events.filter((event) => event.kind === "token").map((event) => event.id);

/**
 * Drops **the time and phase fields** (`wallMs` / `delivered` / `gate`) from a list of observation
 * records.
 *
 * The entry for checks that look only at run kinds and numbers ("one message per run", "cycle
 * numbers are consecutive from 1") so that they do not depend on the wall clock — the
 * non-speculative sequence cannot take a fake clock (`now` is a field of the speculative options),
 * so the real `wallMs` values differ per run. The checks of the wall clock and phases themselves
 * are done separately by the records that take a fake clock.
 */
export const runShape = (phases: readonly GenerationRunPhase[]): GenerationRunPhase[] =>
  phases.map((phase) =>
    phase.kind === "verify"
      ? { kind: phase.kind, cycle: phase.cycle, rows: phase.rows, accepted: phase.accepted }
      : phase.kind === "decode"
        ? { kind: phase.kind, step: phase.step }
        : phase,
  );
