/**
 * The decode bench's data (the record body of the `gemma4-bench` entry): whether the device timed
 * passes, whether the run completed, the workload, the arms with their distributions, every visit
 * and chat (cold ones included), and the summary derived from the chats. The workbench record holds
 * the rest (the entry, its parameters and the environment). Pure, so `vp test` covers it.
 */

import type { SamplerSpec } from "@karume/models/gemma";

import type { DistributionFacts } from "../entry.ts";
import { entryPointOf, type GemmaEntryPoint, type GemmaMount } from "../gemma4/settings.ts";
import {
  type ArmPipelineOptions,
  type ArmSessionOptions,
  type BenchPromptId,
  benchPrompt,
  type BenchSettings,
  PLE_DISTINCT_ID_THRESHOLD,
} from "./plan.ts";
import type { ChatRecord, RunArm, VisitRecord } from "./run.ts";
import { type ArmPromptSummary, summarizeRuns } from "./summary.ts";

/** One arm as the data names it. */
export type ArmRecord = {
  /** 1-based, as the summary names it (arm 1 is the one the summary compares the others with). */
  readonly index: number;
  readonly mount: GemmaMount;
  /** The mount's distribution as the run read it (`openDistribution`'s facts). */
  readonly distribution: DistributionFacts;
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

export type DecodeBenchData = {
  /** Whether the device timed passes (the adapter listed timestamp-query). */
  readonly timestampQuery: boolean;
  /** False when the bench failed or was cancelled (the data holds what finished). */
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

export const buildBenchData = (input: {
  readonly timestampQuery: boolean;
  readonly complete: boolean;
  readonly settings: BenchSettings;
  readonly arms: readonly RunArm[];
  /** The distribution of every mount an arm reads. */
  readonly distributions: Partial<Readonly<Record<GemmaMount, DistributionFacts>>>;
  readonly visits: readonly VisitRecord[];
  readonly chats: readonly ChatRecord[];
}): DecodeBenchData => {
  const { settings, arms, distributions } = input;
  return {
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
        distribution,
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
