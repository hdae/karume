/**
 * The decode bench's record: the environment (with the environment key ADR 0106 names results by),
 * the workload, the arms with their distributions, every visit and chat (cold ones included), and
 * the summary derived from the chats. Pure, so `vp test` covers it.
 */

import type { SamplerSpec } from "@karume/models/gemma";

import type { RecordEnvironment } from "@/lib/environment";
import type { SourceRecord } from "@/lib/record";
import { entryPointOf, type GemmaEntryPoint, type GemmaMount } from "@/models/gemma4/settings";

import {
  type ArmPipelineOptions,
  type ArmSessionOptions,
  type BenchPromptId,
  benchPrompt,
  type BenchSettings,
  PLE_DISTINCT_ID_THRESHOLD,
} from "./plan";
import type { ChatRecord, RunArm, VisitRecord } from "./run";
import { type ArmPromptSummary, summarizeRuns } from "./summary";

/**
 * The version of the saved record; bump it when a field changes meaning or goes away.
 *
 * Version 2: an arm's and a distribution's mount `gemma4-qat` names the q4_0 QAT line (read by
 * `gemma.Pipeline`); the mobile QAT line, which version 1 recorded as `gemma4-qat`, is
 * `gemma4-qat-mobile` (pipeline `gemma4QatMobile.Pipeline`).
 */
export const RECORD_FORMAT = "karume-playground-gemma4-bench/2";

/** Trademark marks dropped from the key's base (`Intel(R) Graphics` → `intel graphics`). */
const TRADEMARKS = ["(r)", "(tm)", "(c)"] as const;

/**
 * The environment key of a Chromium run (ADR 0106: `<runtime>-<adapter name>`): the adapter's
 * `description` when it is filled, else `vendor-architecture` (Chrome fills those two and leaves the
 * description empty), lower-cased, trademark marks dropped, runs of other characters as one `-`.
 * The same rule as the runtime tests' `environmentKey` for the `chrome` runtime. `undefined` outside
 * Chromium (the rule names no other browser) and when the adapter names nothing.
 */
export const chromeEnvironmentKey = (
  userAgent: string,
  info: Pick<GPUAdapterInfo, "vendor" | "architecture" | "description">,
): string | undefined => {
  if (!/\b(?:Chrome|Chromium)\//.test(userAgent)) return undefined;
  const base = info.description !== "" ? info.description : `${info.vendor}-${info.architecture}`;
  let slug = base.toLowerCase();
  for (const mark of TRADEMARKS) slug = slug.replaceAll(mark, "");
  slug = slug.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug === "" ? undefined : `chrome-${slug}`;
};

/** One arm as the record names it. */
export type ArmRecord = {
  /** 1-based, as the page shows it (arm 1 is the one the summary compares the others with). */
  readonly index: number;
  readonly mount: GemmaMount;
  readonly directory: string;
  /** sha256 of the mount's `karume.json`. */
  readonly manifestSha256: string;
  readonly source: SourceRecord;
  readonly pipeline: GemmaEntryPoint;
  readonly model: string;
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  readonly tap: boolean;
  /** Whether the tap ran split (one timed pass per dispatch; the chats carry `tap.perKey`). */
  readonly split: boolean;
  /** Whether the arm ran the integer GEMV (`integerGemv: true` passed explicitly). */
  readonly integerGemv: boolean;
  /**
   * The Session options passed to `fromPretrained` explicitly, on top of the preset's declaration
   * (`{}` = none).
   */
  readonly sessionOptions: ArmSessionOptions;
  /**
   * The pipeline options passed to `fromPretrained` explicitly — `pleResidency` and `decodeAhead`
   * (`{}` = none: the PLE on the host, one decode step per batch).
   */
  readonly pipelineOptions: ArmPipelineOptions;
  readonly sampler: SamplerSpec;
};

export type DecodeBenchRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "gemma4-bench";
  readonly environmentKey?: string;
  /** Whether the device timed passes (the adapter listed timestamp-query). */
  readonly timestampQuery: boolean;
  /** False when the bench failed or was cancelled (the record holds what finished). */
  readonly complete: boolean;
  readonly settings: Omit<BenchSettings, "prompts"> & {
    readonly prompts: readonly {
      readonly id: BenchPromptId;
      readonly text: string;
      readonly distinctIds: string;
    }[];
  };
  readonly arms: readonly ArmRecord[];
  readonly visits: readonly VisitRecord[];
  /** `arm` is 0-based here (the index into `arms`). */
  readonly chats: readonly ChatRecord[];
  /** Over the warm chats; `arm` 0-based. */
  readonly summary: readonly ArmPromptSummary[];
};

export type ArmDistribution = {
  readonly directory: string;
  readonly manifestSha256: string;
};

export const buildBenchRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly timestampQuery: boolean;
  readonly complete: boolean;
  readonly settings: BenchSettings;
  readonly arms: readonly RunArm[];
  /** The distribution of every mount an arm reads. */
  readonly distributions: Partial<Readonly<Record<GemmaMount, ArmDistribution>>>;
  readonly source: SourceRecord;
  readonly visits: readonly VisitRecord[];
  readonly chats: readonly ChatRecord[];
}): DecodeBenchRecord => {
  const { environment, settings, arms, distributions, source } = input;
  const environmentKey = chromeEnvironmentKey(environment.userAgent, environment.adapter);
  return {
    format: RECORD_FORMAT,
    page: "gemma4-bench",
    ...environment,
    ...(environmentKey === undefined ? {} : { environmentKey }),
    timestampQuery: input.timestampQuery,
    complete: input.complete,
    settings: {
      chats: settings.chats,
      newTokens: settings.newTokens,
      rounds: settings.rounds,
      sampler: settings.sampler,
      prompts: settings.prompts.map((id) => {
        const prompt = benchPrompt(id);
        return {
          id,
          text: prompt.text,
          distinctIds: `${prompt.distinctIds === "below" ? "<" : "≥"} ${PLE_DISTINCT_ID_THRESHOLD}`,
        };
      }),
    },
    arms: arms.map((arm, index) => {
      const distribution = distributions[arm.mount];
      if (distribution === undefined) throw new Error(`No distribution facts for '${arm.mount}'`);
      return {
        index: index + 1,
        mount: arm.mount,
        directory: distribution.directory,
        manifestSha256: distribution.manifestSha256,
        source,
        pipeline: entryPointOf(arm.mount),
        model: arm.model,
        preset: { quant: arm.quant, label: arm.quantLabel, isDefault: arm.isDefaultQuant },
        tap: arm.tap,
        split: arm.split === true,
        integerGemv: arm.session?.integerGemv === true,
        sessionOptions: arm.session ?? {},
        pipelineOptions: arm.pipeline ?? {},
        sampler: arm.sampler,
      };
    }),
    visits: input.visits,
    chats: input.chats,
    summary: summarizeRuns(input.chats, arms.length, settings.prompts),
  };
};
