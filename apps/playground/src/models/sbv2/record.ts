/**
 * The saved record of one Style-Bert-VITS2 synthesis and the pure steps that build it: the stage
 * seconds from the per-component marks, the real-time factor and the assembly of the record from the
 * facts of one run ({@link buildSbv2Record}); the page hashes the output (the f32 samples, and the
 * WAV the regression test hashes).
 */

import type { Sbv2RunComponent, Sbv2Utterance } from "@karume/models/sbv2";

import type { PipelineSource } from "@/lib/distributions";
import type { RecordEnvironment } from "@/lib/environment";
import {
  type ComponentRecord,
  seconds,
  type SourceRecord,
  sourceRecord,
  timingOf,
  type TimingTally,
} from "@/lib/record";

import { readingText, type Sbv2Request } from "./request";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-sbv2/1";

/** The three graphs of one synthesis, in the order `generate` runs them. */
export const SBV2_STAGES: readonly Sbv2RunComponent[] = ["text_encoder", "front", "voice"];

export type Sbv2Record = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "sbv2";
  readonly distribution: { readonly directory: string; readonly manifestSha256: string };
  /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
  readonly source: SourceRecord;
  /** The manifest's model (one jvnv speaker each). */
  readonly model: string;
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  /** The text and every option passed to `generate` (all explicit, so the record is the request). */
  readonly request: Sbv2Request;
  /** What the analyser made of the text: the exact utterance `generate` received. */
  readonly analysis: {
    readonly analyzer: string;
    readonly analyzerVersion: string;
    /** The moras with high tones in brackets (`readingText`). */
    readonly reading: string;
    readonly utterance: Sbv2Utterance;
  };
  readonly timings: {
    /**
     * Building the pipeline: the manifest, the descriptors, the download of every weight part into
     * Cache Storage (cached source only — the first build of a model / preset fetches all of its
     * weights; later builds find them there) and the small assets. No Session is created.
     */
    readonly pipelineLoadSeconds: number;
    /** `analyzeText` on the host. */
    readonly analysisSeconds: number;
    /**
     * Seconds per graph, each from the previous mark (the first from the `generate` call). A mark is
     * the graph's diagnostics report, which comes right after its run and before its Session is
     * destroyed, so a stage holds the previous graph's Session destruction, the host glue, its own
     * Session creation (reading and uploading the weights) and its run. The last Session's
     * destruction and the check of the output are in `synthesisSeconds` only.
     */
    readonly stageSeconds: Partial<Record<Sbv2RunComponent, number>>;
    /** The whole `generate` call. */
    readonly synthesisSeconds: number;
    /** Analysis, synthesis and hashing. */
    readonly wallSeconds: number;
  };
  readonly output: {
    readonly sampleRate: number;
    readonly samples: number;
    readonly durationSeconds: number;
    /** `synthesisSeconds / durationSeconds` (below 1 = faster than real time). */
    readonly realTimeFactor: number;
    /** sha256 of the f32 samples as little-endian bytes (what compares across machines). */
    readonly f32Sha256: string;
    /** sha256 of the WAV `encodeWav` makes (16-bit PCM — the hash the regression test pins). */
    readonly wavSha256: string;
  };
  readonly diagnostics: Readonly<Partial<Record<Sbv2RunComponent, ComponentRecord>>>;
};

/**
 * Seconds of each stage from its mark (`performance.now()` when the component's run reported its
 * diagnostics), the first measured from `start`. A stage without a mark (the run failed before it)
 * is left out, and so is every stage after it.
 */
export const stageSeconds = (
  start: number,
  marks: Readonly<Partial<Record<Sbv2RunComponent, number>>>,
): Partial<Record<Sbv2RunComponent, number>> => {
  const result: Partial<Record<Sbv2RunComponent, number>> = {};
  let previous = start;
  for (const stage of SBV2_STAGES) {
    const mark = marks[stage];
    if (mark === undefined) break;
    result[stage] = seconds(mark - previous);
    previous = mark;
  }
  return result;
};

/** Synthesis seconds per second of audio, at 0.001. */
export const realTimeFactor = (synthesisSeconds: number, durationSeconds: number): number => {
  if (!(durationSeconds > 0)) throw new RangeError(`No audio to relate to (${durationSeconds} s)`);
  return Number((synthesisSeconds / durationSeconds).toFixed(3));
};

/** The clock readings of one run (`performance.now()` ms). */
export type Sbv2RunClock = {
  readonly startedAt: number;
  /** After the analysis — the `generate` call starts here. */
  readonly analysedAt: number;
  /** Each graph's diagnostics report. */
  readonly marks: Readonly<Partial<Record<Sbv2RunComponent, number>>>;
  /** `generate` returned. */
  readonly synthesizedAt: number;
  /** The hashes are done. */
  readonly finishedAt: number;
};

/** Everything one finished run knows, as the page collected it. */
export type Sbv2RunFacts = {
  readonly environment: RecordEnvironment;
  readonly distribution: Sbv2Record["distribution"];
  /** The source the pipeline was built from. */
  readonly source: PipelineSource;
  readonly model: string;
  readonly preset: {
    readonly quant: string;
    readonly label: string;
    /** The model's `defaultQuant`. */
    readonly defaultQuant: string;
  };
  readonly pipelineLoadSeconds: number;
  readonly request: Sbv2Request;
  readonly analyzer: { readonly name: string; readonly version: string };
  readonly utterance: Sbv2Utterance;
  readonly clock: Sbv2RunClock;
  readonly audio: { readonly sampleRate: number; readonly samples: number };
  readonly hashes: { readonly f32Sha256: string; readonly wavSha256: string };
  /** Per graph, the folded diagnostics reports (`foldComponentDiagnostics`). */
  readonly diagnostics: ReadonlyMap<Sbv2RunComponent, ComponentRecord>;
  /** Per graph, the GPU time tally; absent for a graph no timed run reported. */
  readonly timings: ReadonlyMap<Sbv2RunComponent, TimingTally>;
};

/**
 * The record of one finished run. The source keeps only its coordinates (`sourceRecord`); a graph's
 * GPU time is added only when it was timed.
 */
export const buildSbv2Record = (facts: Sbv2RunFacts): Sbv2Record => {
  const { clock, audio } = facts;
  const durationSeconds = audio.samples / audio.sampleRate;
  const synthesisSeconds = seconds(clock.synthesizedAt - clock.analysedAt);
  return {
    format: RECORD_FORMAT,
    page: "sbv2",
    ...facts.environment,
    distribution: facts.distribution,
    source: sourceRecord(facts.source),
    model: facts.model,
    preset: {
      quant: facts.preset.quant,
      label: facts.preset.label,
      isDefault: facts.preset.quant === facts.preset.defaultQuant,
    },
    request: facts.request,
    analysis: {
      analyzer: facts.analyzer.name,
      analyzerVersion: facts.analyzer.version,
      reading: readingText(facts.utterance),
      utterance: facts.utterance,
    },
    timings: {
      pipelineLoadSeconds: Number(facts.pipelineLoadSeconds.toFixed(3)),
      analysisSeconds: seconds(clock.analysedAt - clock.startedAt),
      stageSeconds: stageSeconds(clock.analysedAt, clock.marks),
      synthesisSeconds,
      wallSeconds: seconds(clock.finishedAt - clock.startedAt),
    },
    output: {
      sampleRate: audio.sampleRate,
      samples: audio.samples,
      durationSeconds: Number(durationSeconds.toFixed(3)),
      realTimeFactor: realTimeFactor(synthesisSeconds, durationSeconds),
      f32Sha256: facts.hashes.f32Sha256,
      wavSha256: facts.hashes.wavSha256,
    },
    diagnostics: Object.fromEntries(
      [...facts.diagnostics].map(([component, summary]) => {
        const tally = facts.timings.get(component);
        return [component, tally === undefined ? summary : { ...summary, timing: timingOf(tally) }];
      }),
    ),
  };
};
