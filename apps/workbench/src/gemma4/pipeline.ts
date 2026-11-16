/**
 * Builds a Gemma 4 pipeline for a workbench entry on the entry's GPU, from the reference
 * `openDistribution` gave it: `gemma.Pipeline` for the `gemma4` and `gemma4-qat` mounts,
 * `gemma4QatMobile.Pipeline` for `gemma4-qat-mobile`. The workbench's Gemma entries build every
 * pipeline through it (the session A/B's arms and probes, the decode bench's arms), so they load a
 * model the same way. The Playground's chat page keeps a builder of its own.
 *
 * No MTP drafter and no image embedder: the workbench's measurements run neither.
 */

import type { DistributionSource, HubRepoRef } from "@karume/hub";
import { type FromPretrainedOptions, Pipeline, type RunPhase } from "@karume/models/gemma";
import * as gemma4QatMobile from "@karume/models/gemma4-qat-mobile";
import type { GpuContext, SessionDiagnostics } from "@karume/runtime";

import { entryPointOf, type ExpertCacheSlots, type GemmaMount, isQatModel } from "./settings.ts";

/** What a family's `fromPretrained` takes as its first argument. */
export type PretrainedRef = HubRepoRef | DistributionSource;

/** The model a build loads, and the load options that change what is built. */
export type GemmaLoad = {
  readonly mount: GemmaMount;
  readonly model: string;
  readonly quant: string;
  /** Store the full-layer KV in f16 (`fullStateStorage` of `fromPretrained`); absent = f32. */
  readonly fullStateStorage?: "f16";
  /**
   * Keep only `slots` experts of each routed layer on the GPU (`expertCache` of `fromPretrained` —
   * a mixture-of-experts model only), one count for every layer or one per layer; absent = every
   * expert resident. `hostBytes` holds that many bytes of experts in RAM in front of the
   * distribution (`Number.MAX_SAFE_INTEGER` = every expert); absent = every miss reads the
   * distribution. `depth` decodes by self-speculation over the cache; absent = one run per decode
   * token.
   */
  readonly expertCache?: {
    readonly slots: ExpertCacheSlots;
    readonly hostBytes?: number;
    readonly depth?: number;
  };
};

/**
 * Where `onRunDiagnostics` goes. Passing it changes the decode path: the pipeline builds its on-GPU
 * greedy selection only without it, so with it a temperature-0 generation reads the whole logits
 * row back.
 */
export type DiagnosticsSink = (diagnostics: SessionDiagnostics, phase: RunPhase) => void;

/** The explicit Session options a build may pass (the pipeline resolves them over the preset's). */
export type GemmaSessionOptions = Pick<
  FromPretrainedOptions,
  "integerGemv" | "packedStaticQuantize" | "numerics" | "fusionRules"
>;

/** The pipeline options a build may pass as they are. */
export type GemmaPipelineOptions = Pick<FromPretrainedOptions, "pleResidency" | "decodeAhead">;

export type GemmaBuildInput = {
  readonly ref: PretrainedRef;
  readonly gpu: GpuContext;
  readonly load: GemmaLoad;
  /** Aborts the fetch (the construction after it cannot be interrupted). */
  readonly signal: AbortSignal;
  readonly onRunDiagnostics?: DiagnosticsSink;
  readonly onProgress?: FromPretrainedOptions["onProgress"];
  readonly sessionOptions?: GemmaSessionOptions;
  readonly pipelineOptions?: GemmaPipelineOptions;
};

/** The two entry points' `fromPretrained`, injectable so a test sees the options a build passes. */
export type GemmaEntryPoints<Regular, Mobile> = {
  readonly regular: (ref: PretrainedRef, options: FromPretrainedOptions) => Promise<Regular>;
  readonly mobile: (
    ref: PretrainedRef,
    options: gemma4QatMobile.FromPretrainedOptions,
  ) => Promise<Mobile>;
};

export type GemmaPipeline = Pipeline | gemma4QatMobile.Pipeline;

const ENTRY_POINTS: GemmaEntryPoints<Pipeline, gemma4QatMobile.Pipeline> = {
  regular: (ref, options) => Pipeline.fromPretrained(ref, options),
  mobile: (ref, options) => gemma4QatMobile.Pipeline.fromPretrained(ref, options),
};

/**
 * Builds through `entryPoints`: the options every build passes to both entry points, plus the
 * model. A mobile model other than `e2b` / `e4b` is refused here, before any read.
 */
export const buildGemmaPipelineWith = async <Regular, Mobile>(
  entryPoints: GemmaEntryPoints<Regular, Mobile>,
  input: GemmaBuildInput,
): Promise<Regular | Mobile> => {
  const { ref, gpu, load, signal, onRunDiagnostics, onProgress, sessionOptions, pipelineOptions } =
    input;
  const common = {
    ...sessionOptions,
    ...pipelineOptions,
    gpu,
    quant: load.quant,
    ...(load.fullStateStorage === undefined ? {} : { fullStateStorage: load.fullStateStorage }),
    // Both entry points take it; the pipeline itself refuses a graph without experts.
    ...(load.expertCache === undefined ? {} : { expertCache: load.expertCache }),
    signal,
    ...(onRunDiagnostics === undefined ? {} : { onRunDiagnostics }),
    ...(onProgress === undefined ? {} : { onProgress }),
  };
  if (entryPointOf(load.mount) === "gemma.Pipeline") {
    return await entryPoints.regular(ref, { ...common, model: load.model });
  }
  const { model } = load;
  if (!isQatModel(model)) {
    throw new Error(`gemma4QatMobile.Pipeline accepts the models e2b and e4b, not '${model}'`);
  }
  return await entryPoints.mobile(ref, { ...common, model });
};

/** Builds the pipeline `input` names with the two families' own `fromPretrained`. */
export const buildGemmaPipeline = (input: GemmaBuildInput): Promise<GemmaPipeline> =>
  buildGemmaPipelineWith(ENTRY_POINTS, input);
