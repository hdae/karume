/**
 * The session A/B page's arms and the order the runner visits them in.
 *
 * An arm (`SessionAbArm` of `@karume/runtime/tune`) here is one Gemma 4 pipeline configuration:
 * open builds the pipeline with the arm's options (the candidate rule for the `rule` arm), a run is
 * one generation of the request's tokens from the fixed prompt with the fixed sampler — its output
 * is the generated token ids, so the A/B's identity is token identity — the report gives the counts
 * of the run's counted run ({@link countedPhaseOf}, when the request reads them), and close
 * disposes the pipeline.
 *
 * {@link runPositions} is the runner's visiting order as a list, so the page can say where a run is
 * (`runSessionAb` reports no progress of its own); `vp test` drives the real runner with fake arms
 * against it.
 */

import type { ExpertCacheReport, Pipeline, RunPhase } from "@karume/models/gemma";
import type { SessionDiagnostics, SessionOptions } from "@karume/runtime";
import type { SessionAbArm, SessionAbReport, SessionAbRun } from "@karume/runtime/tune";

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
export type ArmRequest = Pick<
  SessionAbRequest,
  "tokens" | "sampler" | "capacity" | "readCounts" | "expertCache"
>;

/**
 * What one arm did, as a caller shows it: `open` before a pipeline is built, `ran` after a run
 * returned (from the report, outside the timed span), `close` before a pipeline is disposed.
 */
export type ArmStep = { readonly arm: string; readonly step: "open" | "ran" | "close" };

/** What an arm uses of a sequence. */
export type ArmSequence = Pick<Awaited<ReturnType<Pipeline["sequence"]>>, "generate" | "dispose">;

/** What an arm uses of a pipeline (a gemma `Pipeline` and the mobile QAT one both have it). */
export type ArmPipeline = {
  readonly sequence: (options: { readonly capacity: number }) => Promise<ArmSequence>;
  readonly expertCacheReport: () => ExpertCacheReport | undefined;
  readonly dispose: () => Promise<void>;
};

/** What an arm reads of a run's diagnostics (`SessionDiagnostics` has both). */
export type CountsDiagnostics = {
  readonly lastRunFusions: Readonly<Record<string, number | undefined>> | undefined;
  readonly lastRunPipelines: SessionDiagnostics["lastRunPipelines"];
};

/** The diagnostics callback a pipeline is built with. */
export type DiagnosticsCallback = (diagnostics: CountsDiagnostics, phase: RunPhase) => void;

/**
 * The run whose counts an arm reads: the generation's first decode run, or under self-speculation
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
  /**
   * The dispatch keys of the arm's rules in its first counted run (absent when the request reads no
   * counts; empty for an arm without rules).
   */
  readonly ruleKeys?: readonly string[];
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

/** One counted run: the runner's counts and the rules' dispatch keys. */
type Counted = { readonly report: SessionAbReport; readonly ruleKeys: readonly string[] };

const countedOf = (diagnostics: CountsDiagnostics, prefixes: readonly string[]): Counted => {
  const fusions: Record<string, number> = {};
  for (const [name, count] of Object.entries(diagnostics.lastRunFusions ?? {})) {
    if (count !== undefined) fusions[name] = count;
  }
  const pipelines = diagnostics.lastRunPipelines ?? [];
  return {
    report: {
      fusions,
      dispatches: pipelines.reduce((sum, { dispatchCount }) => sum + dispatchCount, 0),
    },
    ruleKeys: pipelines
      .map(({ key }) => key)
      .filter((key) => prefixes.some((prefix) => key.startsWith(prefix))),
  };
};

/**
 * One arm on a Gemma 4 pipeline. `build` loads the pipeline with the arm's rules and, when the
 * request reads counts, the diagnostics callback; `prompt` gives the prompt's token ids for a built
 * pipeline; `afterClose` runs after the pipeline's dispose, also when the dispose throws (the page
 * and the e2e test let the device free the pipeline's memory there). The signal is checked before
 * every open and passed to every generation.
 *
 * The timed span (`run`) holds the generation alone: the prompt is encoded when the pipeline is
 * built, and the sequence a run generates in is created before it — by open for the first run, by
 * the report of the run before for the others — and disposed by the next report (or by the
 * pipeline's dispose). The report is always there, so the runner calls it after every run, outside
 * the timed span: it says the run ended (`onStep`), disposes the run's sequence, creates the next
 * one and gives the counts, or none when the request does not read them (the record then has no
 * counts).
 *
 * With counts read, the diagnostics callback is part of every run's generation, timed runs
 * included: the pipeline calls it once per run (and without the expert cache it turns the
 * on-GPU greedy selection and the run-ahead off). The pipeline offers no per-generation switch for
 * it, so it cannot be limited to the untimed runs.
 */
export const pipelineArm = <P extends ArmPipeline>(input: {
  readonly name: string;
  readonly fusionRules?: SessionOptions["fusionRules"];
  readonly request: ArmRequest;
  readonly build: (options: {
    readonly fusionRules?: SessionOptions["fusionRules"];
    readonly onRunDiagnostics?: DiagnosticsCallback;
  }) => Promise<P>;
  readonly prompt: (pipeline: P) => readonly number[];
  readonly afterClose: () => Promise<void>;
  readonly signal: AbortSignal;
  readonly onStep: (step: ArmStep) => void;
}): SessionAbArm & { readonly facts: () => ArmFacts } => {
  const { name, fusionRules, request, signal, onStep } = input;
  const counted = countedPhaseOf(request);
  const prefixes = (fusionRules ?? []).map((rule) => ruleKeyPrefix(rule.name));
  let pipeline: P | undefined;
  let prompt: readonly number[] = [];
  /** The sequence the next run generates in. */
  let next: ArmSequence | undefined;
  /** The sequence the last run generated in, until its report disposes it. */
  let used: ArmSequence | undefined;
  /** The counted run of the run in flight or just returned. */
  let current: Counted | undefined;
  let firstCounted: Counted | undefined;
  let expertCache: ResolvedExpertCache | undefined;
  const onRunDiagnostics: DiagnosticsCallback = (diagnostics, phase) => {
    if (current !== undefined || phase.kind !== counted) return;
    current = countedOf(diagnostics, prefixes);
  };
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
    current = undefined;
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
      pipeline = await input.build({
        ...(fusionRules === undefined ? {} : { fusionRules }),
        ...(request.readCounts ? { onRunDiagnostics } : {}),
      });
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
      if (!request.readCounts) return {};
      if (current === undefined) {
        throw new Error(`Arm ${name}: the run had no ${counted} run to count`);
      }
      firstCounted ??= current;
      return current.report;
    },
    close: async () => {
      onStep({ arm: name, step: "close" });
      const closing = pipeline;
      pipeline = undefined;
      next = undefined;
      used = undefined;
      try {
        // The pipeline's dispose closes the sequences it handed out.
        await closing?.dispose();
      } finally {
        await input.afterClose();
      }
    },
    facts: () => ({
      ...(expertCache === undefined ? {} : { expertCache }),
      ...(firstCounted === undefined ? {} : { ruleKeys: firstCounted.ruleKeys }),
    }),
  };
};

/**
 * One empty submit and its fence, run after an arm's pipeline is disposed. Dawn frees the memory of
 * destroyed buffers only once a submit made after the destroy has completed, so on a device the
 * caller keeps a disposed pipeline stays allocated until the next submit, and the next visit's build
 * of a large model fails out of memory before its first submit (observed 2026-11-11 with Gemma 4
 * 26B under Dawn on Vulkan). Outside the timed span; the pipeline itself is not changed.
 */
export const releaseFence = async (device: GPUDevice): Promise<void> => {
  device.queue.submit([device.createCommandEncoder().finish()]);
  await device.queue.onSubmittedWorkDone();
};
