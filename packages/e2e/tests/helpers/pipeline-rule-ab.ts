/**
 * The shared part of the opt-in end-to-end runs of a candidate fusion rule through a real pipeline
 * (`KARUME_PIPELINE_RULE_AB=<dir>` — e2e_moss_pipeline_rule_test.ts, e2e_anima_pipeline_rule_test.ts,
 * e2e_wan_ti2v_pipeline_rule_test.ts): two pipelines over the same distribution mirror, `base`
 * (without `fusionRules`) and `rule` (the same plus the rule), generate the same request in the
 * order of `runSessionAb` of `@karume/runtime/tune` (an untimed warm-up round, then A B B A per
 * round), and every generation's output is compared word for word with the base's first
 * (`wordIdentity` — f32 with NaN by class, integer words exactly).
 *
 * The arms are pipelines, not Sessions: the timed span is one whole `generate` (text, DiT and VAE
 * stages, each Session built and closed inside it, weights uploaded every time). A pipeline that
 * reports its stages (`onEvent` stage events) also gets the time between each stage's start and
 * end, kept per run next to the record.
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`.
 */

import { mkdir, writeTextFile } from "@karume/test-support/fs";
import {
  judgeSessionAb,
  runSessionAb,
  type SessionAbArm,
  type SessionAbJudgement,
  type SessionAbRecord,
  type SessionAbReport,
  type SessionAbRun,
} from "@karume/runtime/tune";
import { ENVIRONMENT } from "../../../runtime/tests/helpers/environment.ts";

/** The record directory; unset or empty = the runs SKIP (the opt-in). */
export const PIPELINE_RULE_AB_DIRECTORY = process.env["KARUME_PIPELINE_RULE_AB"];

/** The first unmet requirement's reason (`[met, reason]` pairs in order), or undefined. */
export const pipelineRuleAbSkipReason = (
  requirements: readonly (readonly [boolean, string])[],
): string | undefined => {
  if (PIPELINE_RULE_AB_DIRECTORY === undefined || PIPELINE_RULE_AB_DIRECTORY === "") {
    return "KARUME_PIPELINE_RULE_AB is not set (opt-in: KARUME_PIPELINE_RULE_AB=<dir> writes the record there)";
  }
  return requirements.find(([met]) => !met)?.[1];
};

/** Dawn's backend on each platform Node runs it on (the record says the backend is inferred). */
const DAWN_BACKENDS: Readonly<Record<string, string>> = {
  linux: "vulkan",
  darwin: "metal",
  win32: "d3d12",
};

/** The backend Dawn uses on this platform (WebGPU does not expose it). */
export const inferredBackend = (platform: string = process.platform): string => {
  if (!Object.hasOwn(DAWN_BACKENDS, platform)) {
    throw new Error(`No Dawn backend is known for the platform '${platform}'`);
  }
  return DAWN_BACKENDS[platform];
};

/** The stage events of a pipeline (`kind: "stage"`, the component and `start` / `end`). */
type StageEvent = { readonly kind: string; readonly component?: string; readonly at?: string };

/**
 * A stage clock: `observe` takes a pipeline's events, `take` returns the milliseconds between each
 * stage's start and end since the last `take` (a stage that runs twice adds up).
 */
export const stageClock = () => {
  const starts = new Map<string, number>();
  let durations: Record<string, number> = {};
  return {
    observe: (event: StageEvent): void => {
      if (event.kind !== "stage" || event.component === undefined) return;
      if (event.at === "start") starts.set(event.component, performance.now());
      else if (event.at === "end") {
        const start = starts.get(event.component);
        if (start === undefined) return;
        durations[event.component] = (durations[event.component] ?? 0) + performance.now() - start;
      }
    },
    take: (): Readonly<Record<string, number>> => {
      const taken = durations;
      durations = {};
      starts.clear();
      return taken;
    },
  };
};

/** One pipeline arm. */
export type PipelineRuleArm = {
  readonly name: string;
  /** Builds the pipeline (outside the timed span). */
  readonly open: () => Promise<void>;
  /** One generation; returns the outputs to compare (on the host). */
  readonly generate: () => Promise<SessionAbRun["outputs"]>;
  /** The counts of the generation that just returned (outside the timed span). */
  readonly report?: () => Promise<SessionAbReport>;
  /** The stage times of the generation that just returned (absent = the pipeline reports none). */
  readonly stages?: () => Readonly<Record<string, number>>;
  readonly close: () => Promise<void>;
};

/** What {@link runPipelineRuleAb} returns. */
export type PipelineRuleAbResult = {
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
  /** Per arm, every generation's stage times in run order (the warm-up's first). */
  readonly stages: Readonly<Record<string, readonly Readonly<Record<string, number>>[]>>;
  /** The record file written. */
  readonly file: string;
};

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/**
 * Runs the A/B and writes its record (`karume-e2e-pipeline-rule-ab/1`: the request, the rule, the
 * environment, the session A/B record, its judgement and the stage times) as
 * `<dir>/pipeline-rule-ab-<family>_<date>.json`. It asserts nothing: the caller asserts identity and
 * counts; a verdict on times is never asserted (it goes to the record and the log).
 */
export const runPipelineRuleAb = async (input: {
  readonly family: string;
  readonly rule: string;
  readonly request: Readonly<Record<string, unknown>>;
  readonly arms: readonly PipelineRuleArm[];
  readonly rounds: number;
}): Promise<PipelineRuleAbResult> => {
  const stages: Record<string, Readonly<Record<string, number>>[]> = {};
  const arms: SessionAbArm[] = input.arms.map((arm) => {
    stages[arm.name] = [];
    return {
      name: arm.name,
      open: arm.open,
      run: async () => ({ outputs: await arm.generate() }),
      report: async () => {
        // Outside the timed span: the stage times of the run that just returned, then its counts.
        if (arm.stages !== undefined) stages[arm.name].push(arm.stages());
        return arm.report === undefined ? {} : await arm.report();
      },
      close: arm.close,
    };
  });
  const backend = inferredBackend();
  const record = await runSessionAb({
    arms,
    rounds: input.rounds,
    environment: {
      adapter: ENVIRONMENT.adapter,
      userAgent: navigator.userAgent,
      backend,
      date: new Date().toISOString(),
    },
  });
  const judgements = judgeSessionAb(record, { minGain: 0.01 });
  const directory = PIPELINE_RULE_AB_DIRECTORY ?? "";
  await mkdir(directory, { recursive: true });
  const file = `${directory}/pipeline-rule-ab-${input.family}_${record.environment.date.replaceAll(":", "-")}.json`;
  await writeTextFile(
    file,
    `${JSON.stringify(
      {
        format: "karume-e2e-pipeline-rule-ab/1",
        family: input.family,
        rule: input.rule,
        request: input.request,
        environmentKey: ENVIRONMENT.key,
        backendSource: `inferred from process.platform '${process.platform}' (Dawn's backend there)`,
        record,
        judgements,
        stages,
      },
      null,
      2,
    )}\n`,
  );
  console.log(`[pipeline-rule-ab] ${input.family}: record ${file}`);
  for (const arm of record.arms) {
    const times = arm.timesMs.flat();
    const perStage = Object.fromEntries(
      [...new Set(stages[arm.name].flatMap((run) => Object.keys(run)))].map((stage) => [
        stage,
        // The warm-up run is left out of the stage medians, as it is out of the times.
        Number(median(stages[arm.name].slice(1).map((run) => run[stage] ?? 0)).toFixed(1)),
      ]),
    );
    console.log(
      `[pipeline-rule-ab] ${input.family} ${arm.name}: generate median ${median(times).toFixed(1)} ms` +
        ` (${times.map((time) => time.toFixed(0)).join(", ")}), stages ${JSON.stringify(perStage)},` +
        ` counts ${JSON.stringify(arm.fusions ?? {})}, worst identity failures ${arm.identity.worst.failures}`,
    );
  }
  for (const judgement of judgements) {
    console.log(
      `[pipeline-rule-ab] ${input.family} ${judgement.arm}: ${judgement.verdict} — ${judgement.reason}`,
    );
  }
  return { record, judgements, stages, file };
};
