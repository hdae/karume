/**
 * The Irodori-TTS page's run record: the stage order, the real-time factor, the sha256 of the f32
 * samples, and the record itself. Pure (WebCrypto is the only platform API), so `vp test` covers it.
 */

import type { LinearCompute } from "@karume/hub";
import type { RunComponent } from "@karume/models/irodori";

import type { RecordEnvironment } from "@/lib/environment";
import {
  type ComponentRecord,
  f32Sha256,
  intervalSeconds,
  type PipelineCount,
  pipelineOp,
  seconds,
  type SourceRecord,
  type StageTimes,
  stageSecondsOf,
} from "@/lib/record";

import type { ReferenceDecoder } from "./reference";
import type { IrodoriMount, IrodoriRequestParts, LinearChoice } from "./settings";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-irodori/1";

/** The stages in the order a generation runs them (`codec-encoder` only with a reference audio). */
export const STAGE_ORDER: readonly RunComponent[] = [
  "backbone",
  "text-proj",
  "caption-proj",
  "codec-encoder",
  "speaker",
  "duration",
  "dit-context",
  "dit",
  "codec-decoder",
];

/** Synthesis seconds per second of audio (below 1 = faster than real time); null for no audio. */
export const realTimeFactor = (synthesisSeconds: number, audioSeconds: number): number | null =>
  audioSeconds > 0 ? Number((synthesisSeconds / audioSeconds).toFixed(4)) : null;

/** Planned dispatches of linear kernels in a run: the int8-activation route and every other one. */
export type LinearDispatches = { readonly i8a8: number; readonly other: number };

/**
 * The linear dispatches of a component's last run, from its pipeline keys (`linear:v4:i8a8:…`,
 * `linear:v3:f32:…`, `linear_gemv:…`). The key spelling is a runtime diagnostic label outside
 * semver; it is the only place a run shows which linear route it took.
 */
export const linearDispatchesOf = (pipelines: readonly PipelineCount[]): LinearDispatches => {
  let i8a8 = 0;
  let other = 0;
  for (const { key, dispatchCount } of pipelines) {
    if (!pipelineOp(key).startsWith("linear")) continue;
    if (key.split(":")[2] === "i8a8") i8a8 += dispatchCount;
    else other += dispatchCount;
  }
  return { i8a8, other };
};

/** The voice of a request as the record names it. */
export type RecordVoice =
  | {
      readonly kind: "caption";
      /** "" = no voice design (the API's zero caption condition). */
      readonly caption: string;
    }
  | {
      readonly kind: "reference";
      readonly file: string;
      readonly decoder: ReferenceDecoder;
      readonly sampleRate: number;
      readonly samples: number;
      readonly seconds: number;
      /** sha256 of the reference samples that went into the request (little-endian f32). */
      readonly f32Sha256: string;
    };

export type RecordRequest = {
  readonly text: string;
  readonly voice: RecordVoice;
  readonly seed: number;
  readonly durationSeconds: number | "predicted";
  readonly codecTileFrames: number | "default";
  /** Euler steps — fixed by the distribution's `pipelineConfig`, not a request field. */
  readonly steps: number;
};

/** What the record names of the reference file (the samples come from the request). */
export type ReferenceMeta = {
  readonly file: string;
  readonly decoder: ReferenceDecoder;
  readonly f32Sha256: string;
};

export const recordRequest = (
  parts: IrodoriRequestParts,
  reference: ReferenceMeta | undefined,
  steps: number,
): RecordRequest => {
  let voice: RecordVoice;
  if (parts.speaker === undefined) {
    voice = { kind: "caption", caption: parts.caption ?? "" };
  } else {
    if (!("audio" in parts.speaker)) throw new Error("The page sends only a reference audio");
    if (reference === undefined) throw new Error("A reference request needs its file's record");
    const { audio } = parts.speaker;
    voice = {
      kind: "reference",
      file: reference.file,
      decoder: reference.decoder,
      sampleRate: audio.sampleRate,
      samples: audio.data.length,
      seconds: seconds((audio.data.length / audio.sampleRate) * 1000),
      f32Sha256: reference.f32Sha256,
    };
  }
  return {
    text: parts.text,
    voice,
    seed: parts.seed ?? 0,
    durationSeconds: parts.durationSeconds ?? "predicted",
    codecTileFrames: parts.codecTileFrames ?? "default",
    steps,
  };
};

export type IrodoriRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "irodori";
  readonly distribution: {
    readonly mount: IrodoriMount;
    readonly directory: string;
    readonly manifestSha256: string;
  };
  /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
  readonly source: SourceRecord;
  readonly model: string;
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  readonly linearCompute: LinearChoice;
  /** What the DiT Sessions were asked for: the choice, else the preset's declaration, else f32. */
  readonly linearComputeResolved: LinearCompute;
  /**
   * Linear dispatches of the `dit` component's last run (one forward on the host loop, the batch on
   * the resident loop); null when no `dit` diagnostics came. `a8` asked for with `i8a8: 0` = float
   * weights, which the runtime sends to the f32 path.
   */
  readonly ditLinearDispatches: LinearDispatches | null;
  readonly request: RecordRequest;
  /** `host` = one readback per forward (stage events or GPU timing on); `resident` = one batch. */
  readonly ditLoop: "host" | "resident";
  readonly stageEvents: boolean;
  /** Whether this run built the pipeline (and how long that took) or reused the open one. */
  readonly pipeline:
    | { readonly built: true; readonly loadSeconds: number }
    | { readonly built: false };
  /** Empty with stage events off (the resident loop reports no stages). */
  readonly stageSeconds: Partial<Record<RunComponent, number>>;
  readonly timeline: {
    /** Seconds of every denoise step; the first is measured from the `dit` stage start. */
    readonly denoiseStepSeconds: readonly number[];
  };
  /** The `generate` call alone. */
  readonly generateSeconds: number;
  /** From Run to the end of `generate`: the pipeline build (when this run built it) and `generate`. */
  readonly wallSeconds: number;
  readonly output: {
    readonly sampleRate: number;
    readonly samples: number;
    readonly seconds: number;
    /** Latent frames S (the tail trim can make the waveform shorter than S hops). */
    readonly latentFrames: number;
    readonly forwards: number;
    readonly seed: number | null;
    /** `generateSeconds` / `output.seconds`. */
    readonly realTimeFactor: number | null;
    /** sha256 of the output samples (little-endian f32). */
    readonly f32Sha256: string;
  };
  readonly diagnostics: Readonly<Record<string, ComponentRecord>>;
};

export type RecordInput = Pick<
  IrodoriRecord,
  | "distribution"
  | "source"
  | "model"
  | "preset"
  | "linearCompute"
  | "linearComputeResolved"
  | "request"
  | "ditLoop"
  | "stageEvents"
  | "pipeline"
  | "diagnostics"
> & {
  readonly environment: RecordEnvironment;
  /** From the `stage` events (start before a stage's Session, end after). */
  readonly stages: StageTimes<RunComponent>;
  readonly stepMarks: readonly number[];
  /** `performance.now()` marks: the run's start, and the `generate` call's start and end. */
  readonly runStart: number;
  readonly generateStart: number;
  readonly generateEnd: number;
  readonly audio: {
    readonly data: Float32Array<ArrayBuffer>;
    readonly sampleRate: number;
    readonly frames: number;
    readonly forwards: number;
    readonly seed?: number;
  };
};

export const assembleRecord = async (input: RecordInput): Promise<IrodoriRecord> => {
  const { audio, stages } = input;
  const ditPipelines = Object.hasOwn(input.diagnostics, "dit")
    ? input.diagnostics["dit"]?.lastRunPipelines
    : undefined;
  const generateSeconds = seconds(input.generateEnd - input.generateStart);
  const audioSeconds = seconds((audio.data.length / audio.sampleRate) * 1000);
  return {
    format: RECORD_FORMAT,
    page: "irodori",
    ...input.environment,
    distribution: input.distribution,
    source: input.source,
    model: input.model,
    preset: input.preset,
    linearCompute: input.linearCompute,
    linearComputeResolved: input.linearComputeResolved,
    ditLinearDispatches: ditPipelines === undefined ? null : linearDispatchesOf(ditPipelines),
    request: input.request,
    ditLoop: input.ditLoop,
    stageEvents: input.stageEvents,
    pipeline: input.pipeline,
    stageSeconds: stageSecondsOf(stages, STAGE_ORDER),
    timeline: {
      denoiseStepSeconds:
        stages.dit === undefined ? [] : intervalSeconds(stages.dit.start, input.stepMarks),
    },
    generateSeconds,
    wallSeconds: seconds(input.generateEnd - input.runStart),
    output: {
      sampleRate: audio.sampleRate,
      samples: audio.data.length,
      seconds: audioSeconds,
      latentFrames: audio.frames,
      forwards: audio.forwards,
      seed: audio.seed ?? null,
      realTimeFactor: realTimeFactor(generateSeconds, audioSeconds),
      f32Sha256: await f32Sha256(audio.data),
    },
    diagnostics: input.diagnostics,
  };
};
