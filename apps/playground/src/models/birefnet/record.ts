/**
 * The saved record of one BiRefNet run: what ran (checkpoint, model, preset, source), on what
 * (environment), on which input, how long it took, the sha256 of the matte bytes and the matte
 * Session's diagnostics. Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";
import type { SessionDiagnostics } from "@karume/runtime";

import type { PipelineSource } from "@/lib/distributions";
import type { RecordEnvironment } from "@/lib/environment";
import {
  type ComponentRecord,
  foldComponentDiagnostics,
  seconds,
  type SourceRecord,
  sourceRecord,
  tallyTiming,
  timingOf,
} from "@/lib/record";

import type { CheckpointId, Selection } from "./settings";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-birefnet/1";

/** The pipeline's one component (the manifest's weights name). */
const MATTE_COMPONENT = "matte";

export type BirefnetRecord = {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "birefnet";
} & RecordEnvironment & {
    readonly distribution: {
      readonly checkpoint: CheckpointId;
      readonly directory: string;
      readonly manifestSha256: string;
    };
    /** `cached` = the HF-compatible route through the hub's CacheStorage; `direct` = plain fetch. */
    readonly source: SourceRecord;
    /** The manifest's model: the graph's input size (`1024`, `2048`). */
    readonly model: { readonly name: string; readonly isDefault: boolean };
    readonly preset: {
      readonly quant: string;
      readonly label: string;
      readonly isDefault: boolean;
    };
    /** The input as the pipeline saw it: the decoded RGB bytes (alpha dropped) and their sha256. */
    readonly request: {
      readonly image: {
        readonly name: string;
        readonly width: number;
        readonly height: number;
        readonly rgbSha256: string;
      };
    };
    readonly pipeline: {
      /**
       * `false`: this run built the pipeline, so `timings.wallSeconds` holds `loadSeconds` (and the
       * release of the previous pipeline, when there was one).
       */
      readonly reused: boolean;
      /** `fromPretrained` of the pipeline that ran: manifest, weights (network or cache), Session. */
      readonly loadSeconds: number;
      /** 1 for the pipeline's first segmentation (it also compiles the GPU pipelines), then 2, 3… */
      readonly segmentRun: number;
    };
    readonly timings: {
      /** `segment`: preprocess, the one Session run, readback, sigmoid and resize to the input size. */
      readonly segmentSeconds: number;
      /**
       * From Run to the matte: with `pipeline.reused` false, also the release of a previous pipeline
       * and the load.
       */
      readonly wallSeconds: number;
    };
    readonly output: {
      readonly width: number;
      readonly height: number;
      /** sha256 of the matte's bytes (one 8-bit alpha per pixel, row-major, at the input size). */
      readonly matteSha256: string;
    };
    /** `matte`: the Session's report of this run (absent when the run reported none). */
    readonly diagnostics: Readonly<Record<string, ComponentRecord>>;
  };

/** What the record keeps of a Session report (`SessionDiagnostics` satisfies it). */
export type MatteReport = Parameters<typeof foldComponentDiagnostics>[1] & {
  readonly lastRunTiming: SessionDiagnostics["lastRunTiming"];
};

/** One report as a component record, with its GPU time when the device timed the run. */
export const matteDiagnostics = (report: MatteReport): ComponentRecord => {
  const summary = foldComponentDiagnostics(undefined, report);
  const tally = tallyTiming(undefined, report.lastRunTiming);
  return tally === undefined ? summary : { ...summary, timing: timingOf(tally) };
};

export type RecordInput = {
  readonly environment: RecordEnvironment;
  readonly checkpoint: CheckpointId;
  readonly directory: string;
  readonly manifest: Manifest;
  readonly manifestSha256: string;
  readonly source: PipelineSource;
  readonly selection: Selection;
  readonly image: BirefnetRecord["request"]["image"];
  readonly pipeline: {
    readonly reused: boolean;
    readonly loadMs: number;
    readonly segmentRun: number;
  };
  readonly segmentMs: number;
  readonly wallMs: number;
  readonly matte: { readonly width: number; readonly height: number; readonly sha256: string };
  readonly report: MatteReport | undefined;
};

export const assembleRecord = (input: RecordInput): BirefnetRecord => {
  const { manifest, selection } = input;
  if (!Object.hasOwn(manifest.models, selection.model)) {
    throw new Error(`The manifest has no model '${selection.model}'`);
  }
  const entry = manifest.models[selection.model];
  if (!Object.hasOwn(entry.quants, selection.quant)) {
    throw new Error(`Model '${selection.model}' has no preset '${selection.quant}'`);
  }
  const quant = entry.quants[selection.quant];
  return {
    format: RECORD_FORMAT,
    page: "birefnet",
    ...input.environment,
    distribution: {
      checkpoint: input.checkpoint,
      directory: input.directory,
      manifestSha256: input.manifestSha256,
    },
    source: sourceRecord(input.source),
    model: { name: selection.model, isDefault: selection.model === manifest.defaultModel },
    preset: {
      quant: selection.quant,
      label: quant.label ?? selection.quant,
      isDefault: selection.quant === entry.defaultQuant,
    },
    request: { image: input.image },
    pipeline: {
      reused: input.pipeline.reused,
      loadSeconds: seconds(input.pipeline.loadMs),
      segmentRun: input.pipeline.segmentRun,
    },
    timings: { segmentSeconds: seconds(input.segmentMs), wallSeconds: seconds(input.wallMs) },
    output: {
      width: input.matte.width,
      height: input.matte.height,
      matteSha256: input.matte.sha256,
    },
    diagnostics:
      input.report === undefined ? {} : { [MATTE_COMPONENT]: matteDiagnostics(input.report) },
  };
};
