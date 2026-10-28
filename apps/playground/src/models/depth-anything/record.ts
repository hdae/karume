/**
 * The saved record of one depth estimate and the summary the page shows of it. Pure, so `vp test`
 * covers it.
 */

import type { PipelineSource } from "@/lib/distributions";
import type { RecordEnvironment } from "@/lib/environment";
import {
  type ComponentRecord,
  seconds,
  type SourceRecord,
  sourceRecord,
  timingOf,
  type TimingTally,
  topOpShares,
} from "@/lib/record";

import type { DepthRange } from "./colormap";
import type { Size } from "./settings";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-depth-anything/1";

/** The one component of the family (the manifest's weights table names it `depth`). */
export const DEPTH_COMPONENT = "depth";

export type DepthRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "depth-anything";
  readonly distribution: { readonly directory: string; readonly manifestSha256: string };
  readonly source: SourceRecord;
  readonly model: { readonly name: string; readonly isDefault: boolean };
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  readonly request: {
    /** The picked file: its name, size and the sha256 of the RGB8 bytes handed to `estimate`. */
    readonly image: {
      readonly name: string;
      readonly width: number;
      readonly height: number;
      readonly rgbSha256: string;
    };
    /** The square size the graph runs at (the manifest's `pipelineConfig`); the aspect is not kept. */
    readonly modelInput: Size | null;
  };
  readonly timings: {
    /**
     * Seconds to build the pipeline (fetch, checks, Session creation and weight upload) when this
     * run built it; `null` when the run reused the pipeline an earlier run built.
     */
    readonly pipelineLoadSeconds: number | null;
    /** The `estimate` call: host preprocessing, the one `depth` run, the resize back. */
    readonly estimateSeconds: number;
    /** The host work after the estimate: the output range and the two sha256. */
    readonly postprocessSeconds: number;
    readonly wallSeconds: number;
  };
  readonly output: {
    readonly width: number;
    readonly height: number;
    /**
     * The range of the raw relative depth. Every value is finite: `estimate` throws on a non-finite
     * one instead of returning it.
     */
    readonly min: number;
    readonly max: number;
    /** sha256 of the raw f32 depth values (little-endian bytes, row-major, output size). */
    readonly depthSha256: string;
  };
  /**
   * `depth`: the Session diagnostics of this run. `submitCount` / `dispatchCount` are this run's
   * (`perRunSubmitCounts` of the kit). `build` is fixed when the Session is built, so a run that reused
   * the pipeline repeats the build of the run that built it.
   */
  readonly diagnostics: Readonly<Record<string, ComponentRecord>>;
};

/** What one run gathered, beyond the environment. */
export type DepthRunFacts = {
  readonly directory: string;
  readonly manifestSha256: string;
  readonly source: PipelineSource;
  readonly model: string;
  readonly defaultModel: string;
  readonly quant: string;
  readonly quantLabel: string;
  readonly defaultQuant: string;
  readonly image: {
    readonly name: string;
    readonly width: number;
    readonly height: number;
    readonly rgbSha256: string;
  };
  readonly modelInput: Size | undefined;
  /** Milliseconds; `undefined` when the pipeline was reused. */
  readonly pipelineLoadMs: number | undefined;
  readonly estimateMs: number;
  readonly postprocessMs: number;
  readonly wallMs: number;
  readonly output: { readonly width: number; readonly height: number; readonly range: DepthRange };
  readonly depthSha256: string;
  readonly diagnostics: ComponentRecord | undefined;
  readonly timing: TimingTally | undefined;
};

export const buildDepthRecord = (
  environment: RecordEnvironment,
  facts: DepthRunFacts,
): DepthRecord => ({
  format: RECORD_FORMAT,
  page: "depth-anything",
  ...environment,
  distribution: { directory: facts.directory, manifestSha256: facts.manifestSha256 },
  source: sourceRecord(facts.source),
  model: { name: facts.model, isDefault: facts.model === facts.defaultModel },
  preset: {
    quant: facts.quant,
    label: facts.quantLabel,
    isDefault: facts.quant === facts.defaultQuant,
  },
  request: { image: facts.image, modelInput: facts.modelInput ?? null },
  timings: {
    pipelineLoadSeconds: facts.pipelineLoadMs === undefined ? null : seconds(facts.pipelineLoadMs),
    estimateSeconds: seconds(facts.estimateMs),
    postprocessSeconds: seconds(facts.postprocessMs),
    wallSeconds: seconds(facts.wallMs),
  },
  output: {
    width: facts.output.width,
    height: facts.output.height,
    min: facts.output.range.min,
    max: facts.output.range.max,
    depthSha256: facts.depthSha256,
  },
  diagnostics:
    facts.diagnostics === undefined
      ? {}
      : {
          [DEPTH_COMPONENT]:
            facts.timing === undefined
              ? facts.diagnostics
              : { ...facts.diagnostics, timing: timingOf(facts.timing) },
        },
});

const sizeText = ({ width, height }: Size): string => `${width}×${height}`;

/** A depth value in the summary (relative depth has no unit; 4 significant digits read well). */
const depthText = (value: number): string => value.toPrecision(4);

/** How many ops the summary names. */
const TOP_OPS = 5;

/** The rows of the result card's summary (the record keeps the full detail). */
export const summaryRows = (record: DepthRecord): readonly (readonly [string, string])[] => {
  const { request, timings, output } = record;
  const depth = record.diagnostics[DEPTH_COMPONENT];
  const rows: [string, string][] = [
    [
      "Adapter",
      Object.values(record.adapter)
        .filter((value) => value !== "")
        .join(" · "),
    ],
    [
      "Source",
      record.source.kind === "cached"
        ? `cached (${record.source.repo}@${record.source.revision ?? "main"})`
        : "direct",
    ],
    ["Model", `${record.model.name}${record.model.isDefault ? " (default)" : ""}`],
    ["Preset", `${record.preset.quant}${record.preset.isDefault ? " (default)" : ""}`],
    ["Input", `${request.image.name} · ${sizeText(request.image)}`],
    ["Model input", request.modelInput === null ? "—" : sizeText(request.modelInput)],
    ["Output", sizeText(output)],
    [
      "Pipeline load",
      timings.pipelineLoadSeconds === null
        ? "reused (built by an earlier run)"
        : `${timings.pipelineLoadSeconds.toFixed(3)} s`,
    ],
    ["Estimate", `${timings.estimateSeconds.toFixed(3)} s`],
    ["Wall", `${timings.wallSeconds.toFixed(3)} s`],
    ["Depth range", `${depthText(output.min)} … ${depthText(output.max)}`],
    ["Depth sha256 (f32)", output.depthSha256],
    ["Input sha256 (RGB8)", request.image.rgbSha256],
  ];
  if (depth !== undefined) {
    rows.push(
      ["Weights", `${depth.weightsAllocatedGiB} GiB`],
      ["Submits / dispatches (this run)", `${depth.submitCount} / ${depth.dispatchCount}`],
      [
        "Build",
        `${depth.build.partCount} parts · supply wait ${depth.build.supplyWaitMs} ms · upload fence ${depth.build.uploadFenceMs} ms`,
      ],
    );
    if (depth.timing !== undefined) {
      const { timing } = depth;
      rows.push([
        "GPU time",
        `${timing.totalMs} ms · ${topOpShares(timing, TOP_OPS)
          .map(({ op, ms, share }) => `${op} ${ms} ms (${Math.round(share * 100)}%)`)
          .join(", ")}`,
      ]);
    }
  }
  return rows;
};
