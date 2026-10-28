/**
 * The Anima page's run record: the stage order and labels, the DiT residency, and the assembly of
 * the saved JSON from what one run observed. Pure (the page passes the environment and
 * the clock readings in), so `vp test` covers it.
 */

import type {
  AnimaResidency,
  AnimaResidencyAction,
  AnimaResidencyReason,
  AnimaRunComponent,
} from "@karume/models/anima";

import type { RecordEnvironment } from "@/lib/environment";
import {
  type BuildCost,
  type ComponentRecord,
  intervalSeconds,
  seconds,
  type SourceRecord,
  type StageTimes,
  stageSecondsOf,
} from "@/lib/record";

import type { AnimaMount, AnimaRequestParts, EffectiveRequest } from "./settings";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-anima/1";

/** The stages in the order a generation runs them (the components of `stage` events). */
export const STAGES: readonly AnimaRunComponent[] = [
  "text_encoder",
  "text_conditioner",
  "transformer",
  "vae_decoder",
];

export const STAGE_LABELS: Readonly<Record<AnimaRunComponent, string>> = {
  text_encoder: "Text encoder (Qwen3)",
  text_conditioner: "Text conditioner",
  transformer: "Transformer (DiT)",
  vae_decoder: "VAE decoder",
};

/** One `residency` event of the run (ADR 0112), in order. */
export type ResidencyEvent = {
  readonly action: AnimaResidencyAction;
  readonly reason: AnimaResidencyReason;
};

/**
 * Whether the held pipeline holds a resident DiT after a `residency` event: `retained` puts the DiT
 * in the slot, `released` and `evicted` empty it.
 */
export const residentAfter = (action: AnimaResidencyAction): boolean => action === "retained";

/**
 * Whether this run's DiT stage still runs on the DiT carried over from an earlier run, after one
 * more `residency` event. An eviction before the DiT stage ends (`headroom` before the text stages,
 * `out-of-memory` in a text stage or in the DiT stage itself) makes the DiT stage build a new
 * Session; an eviction after it (`headroom` before the VAE) comes too late to change that.
 */
export const carriedOverAfter = (
  carriedOver: boolean,
  action: AnimaResidencyAction,
  transformerEnded: boolean,
): boolean => carriedOver && !(action === "evicted" && !transformerEnded);

/** The saved record of one run (JSON; the page shows a summary of it). */
export type AnimaRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "anima";
  readonly distribution: {
    readonly mount: AnimaMount;
    readonly directory: string;
    readonly manifestSha256: string;
  };
  /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
  readonly source: SourceRecord;
  readonly model: { readonly name: string; readonly isDefault: boolean };
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  /** The fields sent to `generate` (a field left out takes the manifest's default). */
  readonly request: AnimaRequestParts;
  /** The values the run used once the manifest's defaults filled the rest. */
  readonly effective: EffectiveRequest;
  readonly livePreview: boolean;
  /**
   * `reused`: the pipeline opened by an earlier run (same source, model and preset) ran this one;
   * `openSeconds` is then null. Opening reads the manifest and the small assets and, on the cached
   * route, stores the weights in the hub's cache (no Session is created).
   */
  readonly pipeline: { readonly reused: boolean; readonly openSeconds: number | null };
  readonly residency: {
    readonly requested: AnimaResidency;
    readonly events: readonly ResidencyEvent[];
    /** The held pipeline held a DiT kept resident by an earlier run when this run started. */
    readonly residentAtStart: boolean;
    /**
     * The DiT stage ran on that carried-over DiT (no eviction before the stage ended). Then
     * `diagnostics.transformer.build` is the earlier run's build (a Session's build stats are fixed
     * for its lifetime), not a cost of this run.
     */
    readonly transformerBuildCarriedOver: boolean;
  };
  /** Stage start → end; the Session build and release of a stage fall inside it. */
  readonly stageSeconds: Partial<Record<AnimaRunComponent, number>>;
  /**
   * Seconds per denoise step and per VAE tile, in order. The first of each is measured from its
   * stage start, so it also holds the Session creation and the weight upload (unless the DiT was
   * resident from an earlier run).
   */
  readonly timeline: {
    readonly denoiseStepSeconds: readonly number[];
    readonly vaeTileSeconds: readonly number[];
  };
  /** `generate` alone (the pipeline's opening is in `pipeline.openSeconds`). */
  readonly generateSeconds: number;
  readonly output: {
    readonly width: number;
    readonly height: number;
    /** sha256 of the RGBA bytes the pipeline returned. */
    readonly rgbaSha256: string;
  };
  readonly diagnostics: Readonly<Record<string, ComponentRecord>>;
};

/** What one finished run observed (the page collects it; the clock readings are `performance.now()`). */
export type RunObservation = {
  readonly environment: RecordEnvironment;
  readonly mount: AnimaMount;
  readonly directory: string;
  readonly manifestSha256: string;
  readonly source: SourceRecord;
  readonly model: string;
  readonly defaultModel: string;
  readonly quant: string;
  readonly quantLabel: string;
  readonly defaultQuant: string;
  readonly request: AnimaRequestParts;
  readonly effective: EffectiveRequest;
  readonly livePreview: boolean;
  readonly openMilliseconds: number | undefined;
  readonly residencyEvents: readonly ResidencyEvent[];
  readonly residentAtStart: boolean;
  readonly transformerCarriedOver: boolean;
  /**
   * The pipeline emits one start and one end per stage: a stage it re-runs after evicting a resident
   * DiT on an out-of-memory emits no second start, so that stage's clock also covers the failed
   * attempt and the eviction.
   */
  readonly stages: StageTimes<AnimaRunComponent>;
  readonly stepMarks: readonly number[];
  readonly tileMarks: readonly number[];
  readonly generateMilliseconds: number;
  readonly output: { readonly width: number; readonly height: number; readonly sha256: string };
  readonly diagnostics: ReadonlyMap<string, ComponentRecord>;
};

export const assembleRecord = (run: RunObservation): AnimaRecord => ({
  format: RECORD_FORMAT,
  page: "anima",
  ...run.environment,
  distribution: {
    mount: run.mount,
    directory: run.directory,
    manifestSha256: run.manifestSha256,
  },
  source: run.source,
  model: { name: run.model, isDefault: run.model === run.defaultModel },
  preset: { quant: run.quant, label: run.quantLabel, isDefault: run.quant === run.defaultQuant },
  request: run.request,
  effective: run.effective,
  livePreview: run.livePreview,
  pipeline: {
    reused: run.openMilliseconds === undefined,
    openSeconds: run.openMilliseconds === undefined ? null : seconds(run.openMilliseconds),
  },
  residency: {
    requested: run.request.residency,
    events: run.residencyEvents,
    residentAtStart: run.residentAtStart,
    transformerBuildCarriedOver: run.transformerCarriedOver,
  },
  stageSeconds: stageSecondsOf(run.stages, STAGES),
  timeline: {
    denoiseStepSeconds:
      run.stages.transformer === undefined
        ? []
        : intervalSeconds(run.stages.transformer.start, run.stepMarks),
    vaeTileSeconds:
      run.stages.vae_decoder === undefined
        ? []
        : intervalSeconds(run.stages.vae_decoder.start, run.tileMarks),
  },
  generateSeconds: seconds(run.generateMilliseconds),
  output: { width: run.output.width, height: run.output.height, rgbaSha256: run.output.sha256 },
  diagnostics: Object.fromEntries(run.diagnostics),
});

/**
 * The build costs this run paid: every component's latest build, except a DiT carried over from an
 * earlier run (its build happened in that run).
 */
export const buildsOfThisRun = (
  record: Pick<AnimaRecord, "diagnostics" | "residency">,
): readonly BuildCost[] =>
  Object.entries(record.diagnostics)
    .filter(
      ([component]) =>
        !(component === "transformer" && record.residency.transformerBuildCarriedOver),
    )
    .map(([, summary]) => summary.build);
