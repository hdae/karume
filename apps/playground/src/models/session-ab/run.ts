/**
 * The session A/B page's arms and the order the runner visits them in.
 *
 * An arm (`SessionAbArm` of `@karume/runtime/tune`) here is one Gemma 4 pipeline configuration:
 * open builds the pipeline with the arm's options (the candidate rule for the `rule` arm), a run is
 * one generation of the request's tokens from the fixed prompt with the fixed sampler — its output
 * is the generated token ids, so the A/B's identity is token identity — and close disposes the
 * pipeline. The arms are built without the diagnostics callback, so every run is the production
 * path; the fusion and dispatch counts come from a count probe of each arm before the A/B
 * ({@link probeCounts}: one build with the callback, one generation, one dispose).
 *
 * NOTE: the arm builder ({@link pipelineArm} and what it uses, between the twin markers) has a twin
 * in packages/e2e/tests/e2e_gemma4_session_ab_test.ts (the opt-in Node A/B). No module both may
 * import fits the gates and this change's scope (the e2e tests cannot import the playground,
 * `@karume/candidate-rules` holds rules only), so the two copies are kept textually parallel: a
 * change to one MUST be made to the other (tests/session-ab_arm_test.ts compares them).
 *
 * {@link runPositions} is the runner's visiting order as a list, so the page can say where a run is
 * (`runSessionAb` reports no progress of its own); `vp test` drives the real runner with fake arms
 * against it.
 */

import type { ExpertCacheReport, Pipeline, RunPhase } from "@karume/models/gemma";
import type { SessionDiagnostics, SessionOptions } from "@karume/runtime";
import type { SessionAbArm, SessionAbRun } from "@karume/runtime/tune";

import type { SessionAbRequest } from "./request";

/** Where one run sits in the runner's order. */
export type RunPosition = {
  /** 0 = the untimed warm-up round, 1 to `rounds` the timed rounds. */
  readonly round: number;
  /** 1 = the forward pass over the arms, 2 = the reverse pass. */
  readonly visit: 1 | 2;
  readonly arm: number;
  /** 1 or, with `openPerVisit`, 2 (the fresh pipeline's untimed first run comes first). */
  readonly run: 1 | 2;
  /** Whether the runner times this run. */
  readonly timed: boolean;
};

/**
 * Every run in the runner's order: per round the arms forward then in reverse (A B B A), one run
 * per visit, or two with `openPerVisit` (an untimed one on the fresh pipeline, then the timed one).
 */
export const runPositions = (
  arms: number,
  rounds: number,
  openPerVisit: boolean,
): readonly RunPosition[] => {
  const order = Array.from({ length: arms }, (_, arm) => arm);
  const runs: (1 | 2)[] = openPerVisit ? [1, 2] : [1];
  const positions: RunPosition[] = [];
  for (let round = 0; round <= rounds; round += 1) {
    for (const [visit, arm] of [
      ...order.map((arm) => [1, arm] as const),
      ...order.toReversed().map((arm) => [2, arm] as const),
    ]) {
      for (const run of runs) {
        positions.push({ round, visit, arm, run, timed: round > 0 && run === runs.length });
      }
    }
  }
  return positions;
};

/** The request fields an arm reads. */
export type ArmRequest = Pick<SessionAbRequest, "tokens" | "sampler" | "capacity" | "expertCache">;

// ---- The arm builder: twin begin (the same text in e2e_gemma4_session_ab_test.ts, `export` aside) ----

/**
 * What one arm did, as a caller shows it: `open` before a pipeline is built, `ran` after a run
 * returned (from the report, outside the timed span), `close` before a pipeline is disposed.
 */
export type ArmStep = { readonly arm: string; readonly step: "open" | "ran" | "close" };

/**
 * What one count probe did, as a caller shows it: `open` before its pipeline is built, `run` before
 * its one generation, `close` before its pipeline is disposed.
 */
export type ProbeStep = { readonly arm: string; readonly step: "open" | "run" | "close" };

/** What an arm uses of a sequence. */
export type ArmSequence = Pick<Awaited<ReturnType<Pipeline["sequence"]>>, "generate" | "dispose">;

/** What an arm uses of a pipeline (a gemma `Pipeline` and the mobile QAT one both have it). */
export type ArmPipeline = {
  readonly sequence: (options: { readonly capacity: number }) => Promise<ArmSequence>;
  readonly expertCacheReport: () => ExpertCacheReport | undefined;
  readonly dispose: () => Promise<void>;
};

/** What a probe reads of a run's diagnostics (`SessionDiagnostics` has both). */
export type CountsDiagnostics = {
  readonly lastRunFusions: Readonly<Record<string, number | undefined>> | undefined;
  readonly lastRunPipelines: SessionDiagnostics["lastRunPipelines"];
};

/** The diagnostics callback a probe's pipeline is built with. */
export type DiagnosticsCallback = (diagnostics: CountsDiagnostics, phase: RunPhase) => void;

/**
 * The run whose counts a probe reads: the generation's first decode run, or under self-speculation
 * (an expert cache with a depth) its first verify run. A self-speculating decode runs no decode
 * phase: every cycle is one verify run of its drafts plus one row (`depth + 1` rows, fewer near the
 * token limit), and its drafts are not observed. So with a depth the counts are those of a
 * multi-row verify plan, not of a one-row decode plan.
 */
export const countedPhaseOf = (request: Pick<ArmRequest, "expertCache">): "decode" | "verify" =>
  request.expertCache?.depth === undefined ? "decode" : "verify";

/**
 * The prefix of a rule's dispatch keys: an external rule `<namespace>/<rule>` keys every dispatch
 * under `ext:<namespace>:` (ADR 0148 decision 5).
 */
export const ruleKeyPrefix = (ruleName: string): string => {
  const slash = ruleName.indexOf("/");
  if (slash <= 0) throw new Error(`The rule name "${ruleName}" is not <namespace>/<rule>`);
  return `ext:${ruleName.slice(0, slash)}:`;
};

/** The expert cache as an arm's pipeline resolved it. */
export type ResolvedExpertCache = {
  /** Slots per routed layer, in layer order (one count for every layer is spread over them). */
  readonly slots: readonly number[];
  /** The host tier's bytes as passed (the pipeline does not resolve it further). */
  readonly hostBytes?: number;
  /** The self-speculation depth. */
  readonly depth?: number;
};

/** What an arm learned that the runner's record does not keep. */
export type ArmFacts = {
  /** The expert cache as the pipeline resolved it on the arm's first open (absent without one). */
  readonly expertCache?: ResolvedExpertCache;
};

const resolvedExpertCache = (
  report: ExpertCacheReport | undefined,
  hostBytes: number | undefined,
): ResolvedExpertCache | undefined =>
  report === undefined
    ? undefined
    : {
        slots: [...report.slots],
        ...(hostBytes === undefined ? {} : { hostBytes }),
        ...(report.speculation === undefined ? {} : { depth: report.speculation.depth }),
      };

/** One arm's counts, as its probe read them from the counted run ({@link countedPhaseOf}). */
export type ArmCounts = {
  /** The counted run's fusion counts (`lastRunFusions`). */
  readonly fusions: Readonly<Record<string, number>>;
  /** The counted run's planned dispatches (`lastRunPipelines`). */
  readonly dispatches: number;
  /** The dispatch keys of the arm's rules in the counted run (empty for an arm without rules). */
  readonly ruleKeys: readonly string[];
};

const countsOf = (diagnostics: CountsDiagnostics, prefixes: readonly string[]): ArmCounts => {
  const fusions: Record<string, number> = {};
  for (const [name, count] of Object.entries(diagnostics.lastRunFusions ?? {})) {
    if (count !== undefined) fusions[name] = count;
  }
  const pipelines = diagnostics.lastRunPipelines ?? [];
  return {
    fusions,
    dispatches: pipelines.reduce((sum, { dispatchCount }) => sum + dispatchCount, 0),
    ruleKeys: pipelines
      .map(({ key }) => key)
      .filter((key) => prefixes.some((prefix) => key.startsWith(prefix))),
  };
};

/**
 * What an arm and its count probe are built from. `build` loads the pipeline with the options it
 * is given (the arm's rules, and for the probe alone the diagnostics callback); `prompt` gives the
 * prompt's token ids for a built pipeline. The signal is checked before every build and passed to
 * every generation.
 */
export type ArmSpec<P extends ArmPipeline> = {
  readonly name: string;
  readonly fusionRules?: SessionOptions["fusionRules"];
  readonly request: ArmRequest;
  readonly build: (options: {
    readonly fusionRules?: SessionOptions["fusionRules"];
    readonly onRunDiagnostics?: DiagnosticsCallback;
  }) => Promise<P>;
  readonly prompt: (pipeline: P) => readonly number[];
  readonly signal: AbortSignal;
};

/**
 * One arm on a Gemma 4 pipeline, built with the arm's rules and never with the diagnostics
 * callback. The callback is a build-time option of the gemma pipeline, and its presence alone moves
 * every generation of that pipeline onto another decode path (no on-GPU greedy output without the
 * expert cache, no run-ahead, a diagnostics read per run), so an arm built with it would not time
 * the production path. Every run of an arm, timed or not, is the production path; the arm reports
 * no counts to the runner, and the counts come from {@link probeCounts} before the A/B.
 *
 * The timed span (`run`) holds the generation alone: the prompt is encoded when the pipeline is
 * built, and the sequence a run generates in is created before it — by open for the first run, by
 * the report of the run before for the others — and disposed by the next report (or by the
 * pipeline's dispose). The report is always there, so the runner calls it after every run, outside
 * the timed span: it says the run ended (`onStep`), disposes the run's sequence and creates the next
 * one. Close disposes the pipeline, whose dispose resolves once the device has its memory back (ADR
 * 0095, addendum of 2026-11-11), so the next visit's build does not land on top of it.
 */
export const pipelineArm = <P extends ArmPipeline>(
  input: ArmSpec<P> & { readonly onStep: (step: ArmStep) => void },
): SessionAbArm & { readonly facts: () => ArmFacts } => {
  const { name, fusionRules, request, signal, onStep } = input;
  let pipeline: P | undefined;
  let prompt: readonly number[] = [];
  /** The sequence the next run generates in. */
  let next: ArmSequence | undefined;
  /** The sequence the last run generated in, until its report disposes it. */
  let used: ArmSequence | undefined;
  let expertCache: ResolvedExpertCache | undefined;
  const opened = (): P => {
    if (pipeline === undefined) throw new Error(`Arm ${name}: used before open`);
    return pipeline;
  };
  const prepare = async (): Promise<void> => {
    next = await opened().sequence({ capacity: request.capacity });
  };
  const run = async (): Promise<SessionAbRun> => {
    const sequence = next;
    if (sequence === undefined) throw new Error(`Arm ${name}: no sequence prepared for the run`);
    next = undefined;
    used = sequence;
    const ids: number[] = [];
    const stream = sequence.generate({
      prompt,
      maxNewTokens: request.tokens,
      sampler: request.sampler,
      signal,
    });
    for await (const event of stream) if (event.kind === "token") ids.push(event.id);
    await stream.done;
    return { outputs: { tokens: Int32Array.from(ids) } };
  };
  return {
    name,
    open: async () => {
      signal.throwIfAborted();
      onStep({ arm: name, step: "open" });
      pipeline = await input.build(fusionRules === undefined ? {} : { fusionRules });
      // The same options resolve the same way on every open, so the first open's is the arm's.
      expertCache ??= resolvedExpertCache(
        pipeline.expertCacheReport(),
        request.expertCache?.hostBytes,
      );
      prompt = input.prompt(pipeline);
      await prepare();
    },
    run,
    report: async () => {
      onStep({ arm: name, step: "ran" });
      const finished = used;
      used = undefined;
      await finished?.dispose();
      await prepare();
      return {};
    },
    close: async () => {
      onStep({ arm: name, step: "close" });
      const closing = pipeline;
      pipeline = undefined;
      next = undefined;
      used = undefined;
      // The pipeline's dispose closes the sequences it handed out.
      await closing?.dispose();
    },
    facts: () => (expertCache === undefined ? {} : { expertCache }),
  };
};

/**
 * The count probe of one arm, run before the A/B: it builds the arm's pipeline once with the
 * diagnostics callback, runs one generation of the request's prompt and tokens, takes the counts of
 * its counted run ({@link countedPhaseOf}) and disposes the pipeline, also when the generation
 * throws. Its tokens are compared with nothing: the probe exists for the counts only, and nothing of
 * it is timed. Refuses a generation without the counted run.
 */
export const probeCounts = async <P extends ArmPipeline>(
  input: ArmSpec<P> & { readonly onStep: (step: ProbeStep) => void },
): Promise<ArmCounts> => {
  const { name, fusionRules, request, signal, onStep } = input;
  const counted = countedPhaseOf(request);
  const prefixes = (fusionRules ?? []).map((rule) => ruleKeyPrefix(rule.name));
  /** The counted run's counts, once the generation reached it. */
  const seen: { counts?: ArmCounts } = {};
  signal.throwIfAborted();
  onStep({ arm: name, step: "open" });
  const pipeline = await input.build({
    ...(fusionRules === undefined ? {} : { fusionRules }),
    onRunDiagnostics: (diagnostics, phase) => {
      if (seen.counts !== undefined || phase.kind !== counted) return;
      seen.counts = countsOf(diagnostics, prefixes);
    },
  });
  try {
    const sequence = await pipeline.sequence({ capacity: request.capacity });
    onStep({ arm: name, step: "run" });
    const stream = sequence.generate({
      prompt: input.prompt(pipeline),
      maxNewTokens: request.tokens,
      sampler: request.sampler,
      signal,
    });
    for await (const _event of stream) continue;
    await stream.done;
  } finally {
    onStep({ arm: name, step: "close" });
    // The pipeline's dispose closes the sequence it handed out.
    await pipeline.dispose();
  }
  if (seen.counts === undefined) {
    throw new Error(`Arm ${name}: the probe's generation had no ${counted} run to count`);
  }
  return seen.counts;
};

// ---- The arm builder: twin end ----
