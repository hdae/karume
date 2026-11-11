/**
 * The session A/B of `@karume/runtime/tune` on a real decode (opt-in: `KARUME_SESSION_AB=<dir>`):
 * Gemma 4 26B A4B from the distribution mirror at preset `i4`, under the exact expert cache as the
 * 2026-11-10 measurement of the candidate rule had it (64 slots per layer, every expert in the host
 * tier — research 2026-11-10 §8.1), two arms: `base` (no rule) and `rule` (the candidate rule
 * `candidate/gelu_mul` of `@karume/candidate-rules`, passed through the gemma pipeline's
 * experimental `fusionRules`). A run is one greedy generation of `REQUEST.tokens` tokens from one
 * fixed prompt with a fixed seed, and its output is the generated token ids, so the A/B's identity
 * is token identity. Two 26B pipelines do not fit this machine's GPU together, so every visit opens
 * and closes its pipeline (`openPerVisit`).
 *
 * NOTE: the arm builder (between the twin markers) is the same text as the playground's
 * apps/playground/src/models/session-ab/run.ts, `export` aside. No module both may import fits the
 * gates and this change's scope (the e2e tests cannot import the playground,
 * `@karume/candidate-rules` holds rules only), so the two copies are kept textually parallel: a
 * change to one MUST be made to the other (apps/playground/tests/session-ab_arm_test.ts compares
 * them). Its doc says what is inside a timed span and what is counted.
 *
 * Both arms are built on the one device the test keeps, one after the other: an arm's close
 * awaits the pipeline's dispose, which resolves once the device has the memory back.
 *
 * It asserts token identity (every run of both arms equals the base's first) and the rule's hit
 * count in the decode plan (60: the 30 dense and the 30 routed-expert chains — the count of
 * e2e_gemma4_26b_external_rule_test.ts), never a verdict on times: the verdict and its numbers go
 * to the record and the log. The record (`karume-session-ab/1`, inside an envelope with the request)
 * is written under the directory the variable names.
 *
 * Each arm's counts come from `onRunDiagnostics`: the fusion counts and the planned dispatches
 * (`lastRunPipelines`) of the generation's first decode run (no depth is set, so the decode runs no
 * verify). Under the expert cache the callback does not change the decode path (it runs on the
 * host sampler either way), but it runs inside every run, timed ones included.
 *
 * The record's backend is inferred from the platform (linux → vulkan, darwin → metal, win32 →
 * d3d12: Dawn's backend there), and the envelope says so; WebGPU does not expose it.
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable, without the mirror's `26b` model or
 * without an adapter, the case SKIPs with the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import { acquireGpu, type SessionDiagnostics, type SessionOptions } from "@karume/runtime";
import {
  chatPrompt,
  type ExpertCacheReport,
  Pipeline,
  type RunPhase,
  type SamplerSpec,
} from "@karume/models/gemma";
import {
  judgeSessionAb,
  runSessionAb,
  type SessionAbArm,
  type SessionAbReport,
  type SessionAbRun,
} from "@karume/runtime/tune";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  GEMMA4_26B_IN_MIRROR,
  GEMMA4_QAT_MIRROR,
} from "../../models/tests/helpers/gemma-mirror.ts";
import { mirrorDirectory } from "../../models/tests/helpers/mirror.ts";

const DIRECTORY = process.env["KARUME_SESSION_AB"];

const skipReason =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_SESSION_AB is not set (opt-in: KARUME_SESSION_AB=<dir> writes the record there)"
    : !GEMMA4_26B_IN_MIRROR
      ? `${GEMMA4_QAT_MIRROR.pathname} has no 26b model`
      : !GPU_AVAILABLE
        ? "no GPU adapter"
        : undefined;
if (skipReason !== undefined) {
  console.warn(`[karume] the session A/B on Gemma 4 26B A4B SKIPs: ${skipReason}`);
}

/** The request the A/B runs (written into the record's envelope). */
const REQUEST = {
  model: "26b",
  quant: "i4",
  expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER },
  prompt:
    "Explain in detail how a refrigerator keeps food cold, step by step, for a curious teenager.",
  tokens: 96,
  sampler: { temperature: 0, seed: 0 },
  /** The prompt and the tokens fit in it; the expert cache's chunk rows (8) are below it. */
  capacity: 1024,
  rounds: 4,
  openPerVisit: true,
  readCounts: true,
  minGain: 0.01,
} as const;

/** The 30 layers' dense and routed-expert chains, one hit each, in the decode plan. */
const RULE_HITS = 60;

/** Dawn's backend on each platform Node runs it on (the record says the backend is inferred). */
const DAWN_BACKENDS: Readonly<Record<string, string>> = {
  linux: "vulkan",
  darwin: "metal",
  win32: "d3d12",
};

const inferredBackend = (platform: string): string => {
  if (!Object.hasOwn(DAWN_BACKENDS, platform)) {
    throw new Error(`No Dawn backend is known for the platform '${platform}'`);
  }
  return DAWN_BACKENDS[platform];
};

/** The request fields an arm reads. */
type ArmRequest = {
  readonly tokens: number;
  readonly sampler: SamplerSpec;
  readonly capacity: number;
  readonly readCounts: boolean;
  readonly expertCache?: {
    readonly slots: number | readonly number[];
    readonly hostBytes?: number;
    readonly depth?: number;
  };
};

// ---- The arm builder: twin begin (the same text in e2e_gemma4_session_ab_test.ts, `export` aside) ----

/**
 * What one arm did, as a caller shows it: `open` before a pipeline is built, `ran` after a run
 * returned (from the report, outside the timed span), `close` before a pipeline is disposed.
 */
type ArmStep = { readonly arm: string; readonly step: "open" | "ran" | "close" };

/** What an arm uses of a sequence. */
type ArmSequence = Pick<Awaited<ReturnType<Pipeline["sequence"]>>, "generate" | "dispose">;

/** What an arm uses of a pipeline (a gemma `Pipeline` and the mobile QAT one both have it). */
type ArmPipeline = {
  readonly sequence: (options: { readonly capacity: number }) => Promise<ArmSequence>;
  readonly expertCacheReport: () => ExpertCacheReport | undefined;
  readonly dispose: () => Promise<void>;
};

/** What an arm reads of a run's diagnostics (`SessionDiagnostics` has both). */
type CountsDiagnostics = {
  readonly lastRunFusions: Readonly<Record<string, number | undefined>> | undefined;
  readonly lastRunPipelines: SessionDiagnostics["lastRunPipelines"];
};

/** The diagnostics callback a pipeline is built with. */
type DiagnosticsCallback = (diagnostics: CountsDiagnostics, phase: RunPhase) => void;

/**
 * The run whose counts an arm reads: the generation's first decode run, or under self-speculation
 * (an expert cache with a depth) its first verify run. A self-speculating decode runs no decode
 * phase: every cycle is one verify run of its drafts plus one row (`depth + 1` rows, fewer near the
 * token limit), and its drafts are not observed. So with a depth the counts are those of a
 * multi-row verify plan, not of a one-row decode plan.
 */
const countedPhaseOf = (request: Pick<ArmRequest, "expertCache">): "decode" | "verify" =>
  request.expertCache?.depth === undefined ? "decode" : "verify";

/**
 * The prefix of a rule's dispatch keys: an external rule `<namespace>/<rule>` keys every dispatch
 * under `ext:<namespace>:` (ADR 0148 decision 5).
 */
const ruleKeyPrefix = (ruleName: string): string => {
  const slash = ruleName.indexOf("/");
  if (slash <= 0) throw new Error(`The rule name "${ruleName}" is not <namespace>/<rule>`);
  return `ext:${ruleName.slice(0, slash)}:`;
};

/** The expert cache as an arm's pipeline resolved it. */
type ResolvedExpertCache = {
  /** Slots per routed layer, in layer order (one count for every layer is spread over them). */
  readonly slots: readonly number[];
  /** The host tier's bytes as passed (the pipeline does not resolve it further). */
  readonly hostBytes?: number;
  /** The self-speculation depth. */
  readonly depth?: number;
};

/** What an arm learned that the runner's record does not keep. */
type ArmFacts = {
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
 * pipeline. The signal is checked before every open and passed to every generation. Close disposes
 * the pipeline, whose dispose resolves once the device has its memory back (ADR 0095, addendum of
 * 2026-11-11), so the next visit's build does not land on top of it.
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
const pipelineArm = <P extends ArmPipeline>(input: {
  readonly name: string;
  readonly fusionRules?: SessionOptions["fusionRules"];
  readonly request: ArmRequest;
  readonly build: (options: {
    readonly fusionRules?: SessionOptions["fusionRules"];
    readonly onRunDiagnostics?: DiagnosticsCallback;
  }) => Promise<P>;
  readonly prompt: (pipeline: P) => readonly number[];
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
      // The pipeline's dispose closes the sequences it handed out.
      await closing?.dispose();
    },
    facts: () => ({
      ...(expertCache === undefined ? {} : { expertCache }),
      ...(firstCounted === undefined ? {} : { ruleKeys: firstCounted.ruleKeys }),
    }),
  };
};

// ---- The arm builder: twin end ----

describe.skipIf(skipReason !== undefined)("the session A/B on Gemma 4 26B A4B (opt-in)", () => {
  it("the candidate rule keeps every token and takes its 60 chains in the decode plan", async () => {
    const gpu = await acquireGpu();
    try {
      const prompt = [{ role: "user" as const, content: REQUEST.prompt }];
      const armOf = (name: string, fusionRules: SessionOptions["fusionRules"]) =>
        pipelineArm<Pipeline>({
          name,
          ...(fusionRules === undefined ? {} : { fusionRules }),
          request: REQUEST,
          build: (options) =>
            Pipeline.fromPretrained(mirrorDirectory(GEMMA4_QAT_MIRROR), {
              model: REQUEST.model,
              quant: REQUEST.quant,
              gpu,
              expertCache: REQUEST.expertCache,
              ...(options.fusionRules === undefined ? {} : { fusionRules: options.fusionRules }),
              ...(options.onRunDiagnostics === undefined
                ? {}
                : { onRunDiagnostics: options.onRunDiagnostics }),
            }),
          prompt: (pipeline) => chatPrompt(pipeline.tokenizer, prompt),
          signal: new AbortController().signal,
          onStep: () => undefined,
        });
      const backend = inferredBackend(process.platform);
      const record = await runSessionAb({
        arms: [armOf("base", undefined), armOf("rule", [GELU_MUL_RULE])],
        rounds: REQUEST.rounds,
        openPerVisit: REQUEST.openPerVisit,
        environment: {
          adapter: gpu.adapterInfo,
          userAgent: navigator.userAgent,
          backend,
          date: new Date().toISOString(),
        },
      });
      const judgements = judgeSessionAb(record, { minGain: REQUEST.minGain });
      const directory = DIRECTORY ?? "";
      await mkdir(directory, { recursive: true });
      const file = `${directory}/session-ab_${record.environment.date.replaceAll(":", "-")}.json`;
      await writeTextFile(
        file,
        `${JSON.stringify(
          {
            format: "karume-e2e-session-ab/1",
            request: REQUEST,
            backendSource: `inferred from process.platform '${process.platform}' (Dawn's backend there)`,
            record,
            judgements,
          },
          null,
          2,
        )}\n`,
      );
      console.log(`[session-ab] record: ${file}`);
      for (const arm of record.arms) {
        console.log(
          `[session-ab] ${arm.name}: hits ${arm.fusions?.[GELU_MUL_RULE_NAME] ?? 0}, dispatches ${arm.dispatches}, times ${JSON.stringify(arm.timesMs)}`,
        );
      }
      for (const judgement of judgements) {
        console.log(`[session-ab] ${judgement.arm}: ${judgement.verdict} — ${judgement.reason}`);
      }

      const [base, rule] = record.arms;
      assertEquals(
        [base.identity.worst.failures, rule.identity.worst.failures],
        [0, 0],
        "every run's tokens equal the base's first",
      );
      assertEquals(
        [base.fusions?.[GELU_MUL_RULE_NAME] ?? 0, rule.fusions?.[GELU_MUL_RULE_NAME]],
        [0, RULE_HITS],
        "the rule's hits in the decode plan",
      );
    } finally {
      gpu.destroy();
    }
  });
});
