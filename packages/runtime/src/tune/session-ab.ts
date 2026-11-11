/**
 * The wall-clock A/B of whole runs (`runSessionAb` of `@karume/runtime/tune`): two or more arms —
 * Session configurations, or pipelines a caller wraps — run the same work in alternating order, and
 * the record keeps every run's wall-clock time and whether every run's outputs are u32-identical to
 * the first arm's. It is how a fusion rule (built-in or external) is accepted or dropped: on the
 * measured end-to-end time of untimed runs and on the identity of the outputs, not on per-dispatch
 * GPU time, which overstates what removing a small dispatch saves (research 2026-11-10 §8.1). The
 * statistics and the verdict are pure functions over the record (session-ab-judge.ts).
 *
 * It shares nothing with the geometry sweep: no case list, no fingerprint, its own record format
 * (`karume-session-ab/1`). A case added here moves no geometry profile's fingerprint.
 *
 * ## The protocol
 *
 * - One untimed warm-up round first: pipelines are compiled and plans are derived on the first run.
 * - Each round visits the arms in order and then in reverse order (A B B A for two arms, A B C C B A
 *   for three), so every arm is visited twice per round and a drift that is linear within the round
 *   adds the same amount to every arm's mean of its two visits — for any number of arms. A cost of
 *   switching from one arm to another (caches, pipelines, residency) is balanced only for two arms:
 *   with three or more, a middle arm follows another arm twice per round and the outer arms once.
 * - The timed span is the arm's `run` alone, which includes the readback of the outputs it returns.
 *   Opening and closing an arm are never inside it, in either opening mode: every arm opened once
 *   before the warm-up and closed at the end (the default), or opened and closed around every visit
 *   (`openPerVisit`, for models whose Sessions do not fit in memory together). With `openPerVisit` a
 *   visit is open, one untimed run, the timed run, close: the first run of a fresh Session derives
 *   its plan, which is not what the A/B measures.
 * - The arm's counts (fusion hits, dispatches) are read by its `report`, which the runner calls after
 *   every run, outside the timed span. Every run must report the counts of the arm's first run: a
 *   plan that changes between runs makes the A/B meaningless, so the runner refuses it.
 * - Every run's outputs, the warm-up's and the untimed runs' included, are compared with the base
 *   arm's (the first arm's) first outputs, copied when they are captured, through `wordIdentity`:
 *   f32 words with NaN by class, integer words exactly, and an output of another array type fails
 *   whole. The base arm is compared with its own first run too, so a base that is not deterministic
 *   shows in its own record. A base whose first run returns no word at all is refused: an identity
 *   over nothing proves nothing.
 *
 * MUST: never run it while other work uses the same GPU — the times would measure the contention.
 * The runtime cannot see other work on the device; that is the caller's responsibility, as for the
 * geometry sweep.
 */

import { type NanPolicy, type WordMismatch, wordIdentity } from "./word-identity.ts";

/** The record format of {@link runSessionAb}. */
export const SESSION_AB_FORMAT = "karume-session-ab/1";

/**
 * One output of an arm's run: the words of a 4-byte element array. A `Float32Array` is compared
 * with NaN by class (a computed NaN may change its payload); an `Int32Array` or a `Uint32Array`
 * (token ids, a bool mask) word for word.
 */
export type SessionAbOutput =
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint32Array<ArrayBuffer>;

/** What one run of an arm returns. */
export type SessionAbRun = {
  /** The run's outputs by name, read back to the host inside the run. */
  readonly outputs: Readonly<Record<string, SessionAbOutput>>;
};

/** The counts of the arm's latest run, as its `report` reads them. */
export type SessionAbReport = {
  /** The fusion hit counts of the run, when the arm can read them (per rule or counter name). */
  readonly fusions?: Readonly<Record<string, number>>;
  /** The number of dispatches the run recorded, when the arm can read it. */
  readonly dispatches?: number;
};

/**
 * One side of the A/B: a value with three operations and an optional fourth. The runner calls
 * `open` before the arm's first run (once, or before every visit with `openPerVisit`), `run` for
 * every run, `report` (when present) after every run once the clock has stopped, and `close` after
 * its last run, also when a run throws. It never looks inside an arm: `sessionArm` builds one from a
 * prepared model, and a caller with a pipeline (an LLM decode) passes its own.
 */
export type SessionAbArm = {
  /** The arm's name in the record; unique within one A/B. */
  readonly name: string;
  readonly open: () => Promise<void>;
  /**
   * One run; it MUST return only after the outputs are on the host (the timed span ends there), and
   * SHOULD do nothing else: whatever it does is timed.
   */
  readonly run: () => Promise<SessionAbRun>;
  /** The counts of the run that just returned, read outside the timed span. */
  readonly report?: () => Promise<SessionAbReport>;
  readonly close: () => Promise<void>;
};

/**
 * The adapter's four fields, as the caller read them. A `GPUAdapterInfo` (`GpuContext.adapterInfo`)
 * can be passed as it is: the record copies the four fields by name.
 */
export type SessionAbAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/**
 * Where the A/B ran, as the caller states it. The runner reads none of it: WebGPU does not expose
 * the backend, and the runner reads no clock for dates.
 */
export type SessionAbEnvironment = {
  readonly adapter: SessionAbAdapter;
  /** `navigator.userAgent`, or the runtime's name and version outside a browser. */
  readonly userAgent: string;
  /** The backend's name (`"vulkan"`, `"d3d12"`, `"metal"`, ...). */
  readonly backend: string;
  /** When the A/B ran (ISO 8601). */
  readonly date: string;
};

/** The options of {@link runSessionAb}. */
export type SessionAbOptions = {
  /** Two or more arms; the first is the base the others are compared with. Names are unique. */
  readonly arms: readonly SessionAbArm[];
  /** The timed rounds, an integer of at least 3 (the warm-up round is extra). */
  readonly rounds: number;
  readonly environment: SessionAbEnvironment;
  /**
   * Open and close an arm around every visit, warm-up included, instead of once for the whole A/B
   * — for a model whose Sessions do not fit in memory together. A visit is then open, one untimed
   * run (the fresh Session derives its plan and params there), the timed run, close.
   */
  readonly openPerVisit?: boolean;
  /** The clock in milliseconds. Defaults to `performance.now`. */
  readonly clock?: () => number;
};

/** The causes of a {@link SessionAbMismatch}. */
export const SESSION_AB_MISMATCH_CAUSES = ["word", "length", "type", "missing", "extra"] as const;

/** The first failing word of a run, with the output it belongs to. */
export type SessionAbMismatch = WordMismatch & {
  readonly output: string;
  /**
   * `"word"`: the words differ. `"length"`: the output is shorter or longer than the base's (`index`
   * is the first index past the shorter one, 0 standing for the absent word). `"type"`: the output
   * is another typed-array type than the base's (it fails whole; `index` 0). `"missing"`: the base
   * has the output and this run does not. `"extra"`: this run has an output the base does not.
   */
  readonly cause: (typeof SESSION_AB_MISMATCH_CAUSES)[number];
};

/** The identity of one run's outputs against the base arm's first outputs. */
export type SessionAbVisitIdentity = {
  /** The round: 0 for the warm-up, 1 to `rounds` for the timed rounds. */
  readonly round: number;
  /** The visit within the round: 1 for the forward pass, 2 for the reverse pass. */
  readonly visit: 1 | 2;
  /**
   * The run within the visit: 1 for the visit's only run, or with `openPerVisit` for the untimed
   * first run of the fresh Session; 2 for the run that follows it with `openPerVisit` (the timed
   * one, outside the warm-up round).
   */
  readonly run: 1 | 2;
  /** The words of the base's outputs, every output summed. */
  readonly words: number;
  /**
   * The failing words, every output summed (a missing or extra output, or one of another type,
   * counts all its words).
   */
  readonly failures: number;
  /** The first failing word, present exactly when `failures > 0`. */
  readonly first?: SessionAbMismatch;
};

/**
 * One arm's identity over all its runs. The arm was u32-identical to the base's first outputs
 * exactly when `worst.failures` is 0.
 */
export type SessionAbIdentity = {
  /**
   * The number of runs compared: two visits per round, the warm-up round's included, and two runs
   * per visit with `openPerVisit`.
   */
  readonly runs: number;
  /** The run with the most failing words (the earliest among equals). */
  readonly worst: SessionAbVisitIdentity;
};

/** One arm in the record. */
export type SessionAbArmRecord = {
  readonly name: string;
  /** The fusion counts every run of the arm reported, when it reports them. */
  readonly fusions?: Readonly<Record<string, number>>;
  /** The dispatch count every run of the arm reported, when it reports it. */
  readonly dispatches?: number;
  /** Every timed run in milliseconds: `timesMs[r][v]` is round `r + 1`, visit `v + 1`. */
  readonly timesMs: readonly (readonly [number, number])[];
  readonly identity: SessionAbIdentity;
};

/** The record of {@link runSessionAb} (`karume-session-ab/1`), plain JSON data. */
export type SessionAbRecord = {
  readonly format: typeof SESSION_AB_FORMAT;
  readonly environment: SessionAbEnvironment;
  readonly rounds: number;
  readonly openPerVisit: boolean;
  /** The arms in the order given; the first is the base. */
  readonly arms: readonly SessionAbArmRecord[];
};

/** Refuses what {@link runSessionAb} and the judge cannot work with: arms and rounds. */
export const assertSessionAbShape = (names: readonly string[], rounds: number): void => {
  if (names.length < 2) {
    throw new RangeError(`session A/B: needs at least two arms (got ${names.length})`);
  }
  const seen = new Set<string>();
  for (const name of names) {
    if (seen.has(name)) throw new RangeError(`session A/B: the arm name "${name}" is used twice`);
    seen.add(name);
  }
  if (!Number.isInteger(rounds) || rounds < 3) {
    throw new RangeError(`session A/B: rounds must be an integer of at least 3 (got ${rounds})`);
  }
};

/** The refusal of times that are not finite and non-negative, worded alike by runner and judge. */
export const invalidTimesMessage = (arm: string, rounds: number): string =>
  `session A/B: arm "${arm}" does not hold two finite non-negative times for each of the ${rounds}` +
  ` rounds`;

const nanPolicyOf = (output: SessionAbOutput): NanPolicy =>
  output instanceof Float32Array ? "class" : "payload";

const wordsOf = (output: SessionAbOutput): Uint32Array<ArrayBuffer> =>
  new Uint32Array(output.buffer, output.byteOffset, output.length);

/** Compares one run's outputs with the base's first outputs, output by output. */
const runIdentity = (
  outputs: SessionAbRun["outputs"],
  base: SessionAbRun["outputs"],
  where: Pick<SessionAbVisitIdentity, "round" | "visit" | "run">,
): SessionAbVisitIdentity => {
  let words = 0;
  let failures = 0;
  let first: SessionAbMismatch | undefined;
  for (const [output, expected] of Object.entries(base)) {
    words += expected.length;
    if (!Object.hasOwn(outputs, output)) {
      failures += Math.max(expected.length, 1);
      first ??= {
        output,
        cause: "missing",
        index: 0,
        expected: wordsOf(expected)[0] ?? 0,
        actual: 0,
      };
      continue;
    }
    const actual = outputs[output];
    if (actual.constructor !== expected.constructor) {
      // The same bits under another element type are another value (an f32 is not an i32).
      failures += Math.max(expected.length, actual.length, 1);
      first ??= {
        output,
        cause: "type",
        index: 0,
        expected: wordsOf(expected)[0] ?? 0,
        actual: wordsOf(actual)[0] ?? 0,
      };
      continue;
    }
    const identity = wordIdentity(wordsOf(actual), wordsOf(expected), nanPolicyOf(expected));
    failures += identity.failures;
    if (identity.first !== undefined && first === undefined) {
      const lengthOnly = identity.first.index >= Math.min(actual.length, expected.length);
      first = { output, cause: lengthOnly ? "length" : "word", ...identity.first };
    }
  }
  for (const [output, actual] of Object.entries(outputs)) {
    if (Object.hasOwn(base, output)) continue;
    failures += Math.max(actual.length, 1);
    first ??= { output, cause: "extra", index: 0, expected: 0, actual: wordsOf(actual)[0] ?? 0 };
  }
  return first === undefined ? { ...where, words, failures } : { ...where, words, failures, first };
};

/** A copy of a run's outputs, so a caller that reuses its arrays cannot change the reference. */
const copyOutputs = (outputs: SessionAbRun["outputs"]): SessionAbRun["outputs"] =>
  Object.fromEntries(Object.entries(outputs).map(([name, output]) => [name, output.slice()]));

/** A copy of a report (the record keeps no array or object the arm still holds). */
const copyReport = (report: SessionAbReport): SessionAbReport => ({
  ...(report.fusions === undefined ? {} : { fusions: { ...report.fusions } }),
  ...(report.dispatches === undefined ? {} : { dispatches: report.dispatches }),
});

const sameReport = (a: SessionAbReport, b: SessionAbReport): boolean => {
  if (a.dispatches !== b.dispatches) return false;
  const left = a.fusions;
  const right = b.fusions;
  if (left === undefined || right === undefined) return left === right;
  const names = Object.keys(left);
  return (
    names.length === Object.keys(right).length &&
    names.every((name) => Object.hasOwn(right, name) && left[name] === right[name])
  );
};

/** One arm's state while the A/B runs. */
type ArmState = {
  readonly arm: SessionAbArm;
  readonly times: [number, number][];
  /** The counts of the arm's first run, which every later run must report again. */
  report?: SessionAbReport;
  /** Every run's identity, in run order. */
  readonly identities: SessionAbVisitIdentity[];
};

/**
 * Closes every arm in `opened`, in reverse order of opening. Every close is attempted; the first
 * close failure is thrown after the loop unless `failing` (an error already propagating) is set, in
 * which case that error wins and is rethrown by the caller.
 */
const closeAll = async (opened: readonly SessionAbArm[], failing: boolean): Promise<void> => {
  let closeError: { readonly error: unknown } | undefined;
  for (const arm of opened.toReversed()) {
    try {
      await arm.close();
    } catch (error) {
      closeError ??= { error };
    }
  }
  if (closeError !== undefined && !failing) throw closeError.error;
};

/**
 * Runs the A/B and returns its record (`karume-session-ab/1`). Refuses (`RangeError`) fewer than two
 * arms, a name used twice and rounds that are not an integer of at least 3 before any arm opens;
 * then a base whose first run returns no word, a run whose counts differ from its arm's first run's,
 * and a timed span the clock reads as not finite or negative. A refusal, or an open, a run or a
 * report that throws, ends the A/B: every open arm is closed and the error propagates as it is.
 */
export const runSessionAb = async (options: SessionAbOptions): Promise<SessionAbRecord> => {
  const { arms, rounds, environment } = options;
  assertSessionAbShape(
    arms.map((arm) => arm.name),
    rounds,
  );
  const clock = options.clock ?? (() => performance.now());
  const openPerVisit = options.openPerVisit ?? false;
  const states: ArmState[] = arms.map((arm) => ({ arm, times: [], identities: [] }));
  let baseOutputs: SessionAbRun["outputs"] | undefined;

  /** What follows every run, outside the timed span: the arm's report, then the identity. */
  const check = async (
    state: ArmState,
    run: SessionAbRun,
    where: Pick<SessionAbVisitIdentity, "round" | "visit" | "run">,
  ): Promise<void> => {
    const { arm } = state;
    if (arm.report !== undefined) {
      const report = copyReport(await arm.report());
      if (state.report === undefined) state.report = report;
      else if (!sameReport(report, state.report)) {
        throw new RangeError(
          `session A/B: arm "${arm.name}" reported ${JSON.stringify(report)} in round` +
            ` ${where.round}, visit ${where.visit}, run ${where.run}, but` +
            ` ${JSON.stringify(state.report)} on its first run; a plan that changes between runs` +
            ` makes the A/B meaningless`,
        );
      }
    }
    if (baseOutputs === undefined) {
      const first = copyOutputs(run.outputs);
      if (Object.values(first).every((output) => output.length === 0)) {
        throw new RangeError(
          `session A/B: the base arm "${arm.name}" returned no output word on its first run; an` +
            ` identity over nothing proves nothing`,
        );
      }
      baseOutputs = first;
    }
    state.identities.push(runIdentity(run.outputs, baseOutputs, where));
  };

  /**
   * One visit: open (per-visit mode), the untimed first run of the fresh Session (per-visit mode),
   * the timed run, close (per-visit mode). Every run is checked after it returns.
   */
  const visit = async (state: ArmState, round: number, slot: 1 | 2): Promise<void> => {
    const { arm } = state;
    if (openPerVisit) await arm.open();
    let failed = true;
    try {
      if (openPerVisit) await check(state, await arm.run(), { round, visit: slot, run: 1 });
      const start = clock();
      const run = await arm.run();
      const elapsed = clock() - start;
      if (round > 0) {
        if (!(Number.isFinite(elapsed) && elapsed >= 0)) {
          throw new RangeError(
            `${invalidTimesMessage(arm.name, rounds)} (round ${round}, visit ${slot}: ${elapsed} ms)`,
          );
        }
        if (slot === 1) state.times.push([elapsed, Number.NaN]);
        else state.times[round - 1][1] = elapsed;
      }
      await check(state, run, { round, visit: slot, run: openPerVisit ? 2 : 1 });
      failed = false;
    } finally {
      if (openPerVisit) await closeAll([arm], failed);
    }
  };

  const opened: SessionAbArm[] = [];
  let failing = true;
  try {
    if (!openPerVisit) {
      for (const arm of arms) {
        await arm.open();
        opened.push(arm);
      }
    }
    for (let round = 0; round <= rounds; round += 1) {
      for (const state of states) await visit(state, round, 1);
      for (const state of states.toReversed()) await visit(state, round, 2);
    }
    failing = false;
  } finally {
    await closeAll(opened, failing);
  }

  return {
    format: SESSION_AB_FORMAT,
    environment: {
      // Field by field: a browser's GPUAdapterInfo holds its fields as accessors on the prototype,
      // which an object spread does not copy.
      adapter: {
        vendor: environment.adapter.vendor,
        architecture: environment.adapter.architecture,
        device: environment.adapter.device,
        description: environment.adapter.description,
      },
      userAgent: environment.userAgent,
      backend: environment.backend,
      date: environment.date,
    },
    rounds,
    openPerVisit,
    arms: states.map(armRecord),
  };
};

const armRecord = (state: ArmState): SessionAbArmRecord => ({
  name: state.arm.name,
  ...state.report,
  timesMs: state.times.map(([first, second]) => [first, second] as const),
  identity: {
    runs: state.identities.length,
    // The earliest of the runs with the most failing words; every arm ran at least eight times.
    worst: state.identities.reduce((worst, identity) =>
      identity.failures > worst.failures ? identity : worst,
    ),
  },
});
