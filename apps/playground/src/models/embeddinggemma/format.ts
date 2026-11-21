/**
 * The record as the page shows it: the build status line, the rows table, the similarity matrix and
 * the summary rows. Pure, so `vp test` covers it.
 */

import type { AssetProgress } from "@karume/hub";

import type { RunProgress } from "@/hooks/use-run-task";
import { type BuildCost, type ComponentTiming, seconds, topOpShares } from "@/lib/record";

import type { EmbeddingGemmaRecord, RowOutputRecord } from "./record";

/** How many ops the summary names when the run was timed. */
const TOP_OPS = 5;

const GIB = 2 ** 30;
const MIB = 2 ** 20;

const mib = (bytes: number): string => (bytes / MIB).toFixed(0);

/** What a progress event of `fromPretrained` carries (an `AssetProgress` satisfies it). */
export type BuildProgress = Pick<
  AssetProgress,
  "phase" | "path" | "loaded" | "total" | "fileLoaded" | "fileTotal"
>;

/**
 * The status line of a build from its latest progress event. Once `loaded` reaches `total` every
 * file is fetched and `fromPretrained` goes on to build the Session (reading the weight parts and
 * uploading them to the GPU), which takes no AbortSignal and streams no progress: its own stage,
 * not cancellable.
 */
export const buildStatus = (progress: BuildProgress): RunProgress =>
  progress.loaded >= progress.total
    ? {
        stage: "building the Session",
        note: `uploading the weights (${mib(progress.total)} MiB fetched)`,
        cancellable: false,
        cancelNote: "The Session build takes no signal: Cancel is unavailable until it ends.",
      }
    : {
        stage: "fetching the weights",
        note: `${progress.phase} ${progress.path} (${mib(progress.fileLoaded)} / ${mib(progress.fileTotal)} MiB) · total ${mib(progress.loaded)} / ${mib(progress.total)} MiB`,
      };

/** `Row 1`, … (1-based, as the form numbers them). */
export const rowLabel = (index: number): string => `Row ${index + 1}`;

/** A similarity cell: the value at 4 decimals, or "—" when there is none. */
export const similarityText = (value: number | null): string =>
  value === null ? "—" : value.toFixed(4);

/** What one row gave, in one line: the width and the norm of its vector, or "refused". */
export const rowOutcomeText = (output: RowOutputRecord): string => {
  if (output.kind === "refused") return "refused";
  const norm = output.l2Norm === null ? `${output.nonFinite} non-finite` : `norm ${output.l2Norm}`;
  return `${output.dimension} values · ${norm} · ${output.embedSeconds} s`;
};

/** The first values of a row's vector, at 4 decimals ("" for a refused row). */
export const firstValuesText = (output: RowOutputRecord): string =>
  output.kind === "refused"
    ? ""
    : output.first
        .map((value) => (typeof value === "number" ? value.toFixed(4) : value))
        .join(", ");

const buildText = (cost: BuildCost): string =>
  [
    `supply wait ${seconds(cost.supplyWaitMs)} s`,
    `decode ${seconds(cost.decodeMs)} s`,
    `buffer create ${seconds(cost.bufferCreateMs)} s`,
    `write issue ${seconds(cost.writeBufferIssueMs)} s`,
    `upload fence ${seconds(cost.uploadFenceMs)} s`,
    `${(cost.uploadedBytes / GIB).toFixed(2)} GiB in ${cost.partCount} parts`,
  ].join(" · ");

const timingText = (timing: ComponentTiming): string =>
  [
    `${timing.totalMs.toFixed(1)} ms timed over ${timing.runs} runs`,
    ...topOpShares(timing, TOP_OPS).map(
      ({ op, ms, share }) => `${op} ${ms.toFixed(1)} ms ${(share * 100).toFixed(1)}%`,
    ),
    ...(timing.clampedNegativeSamples === 0
      ? []
      : [`${timing.clampedNegativeSamples} samples clamped to 0`]),
  ].join(" · ");

/** The summary rows of the result card (the JSON keeps the full detail). */
export const summaryRows = (
  record: EmbeddingGemmaRecord,
): readonly (readonly [string, string])[] => {
  const { timings, diagnostics } = record;
  const model = diagnostics.model;
  const embedded = record.output.rows.filter(({ kind }) => kind === "embedding").length;
  return [
    [
      "Adapter",
      `${record.adapter.vendor} · ${record.adapter.architecture} · ${record.adapter.device}`,
    ],
    ["Driver", record.adapter.description],
    [
      "Device",
      `subgroups ${record.device.features.includes("subgroups")} · GPU timing ${record.device.gpuTimingEnabled}`,
    ],
    [
      "Source",
      record.source.kind === "cached"
        ? `cached (${record.source.repo} @ ${record.source.revision?.slice(0, 12) ?? "main"})`
        : "direct",
    ],
    ["Model", `${record.model.name}${record.model.isDefault ? " (default)" : ""}`],
    ["Preset", `${record.preset.quant}${record.preset.isDefault ? " (default)" : ""}`],
    ["Dimension", String(record.request.dimension)],
    ["Rows", `${record.output.rows.length} (${embedded} embedded)`],
    [
      "Build s",
      timings.buildSeconds === null
        ? "— (pipeline reused from an earlier run)"
        : String(timings.buildSeconds),
    ],
    ["Wall s", String(timings.wallSeconds)],
    ...(model === undefined
      ? []
      : [
          [
            "Model Session",
            `${model.runs} runs · weights ${model.weightsAllocatedGiB} GiB · ${model.submitCount} submits · ${model.dispatchCount} dispatches (this run; Session total ${model.sessionSubmitCount} / ${model.sessionDispatchCount}) · geometry ${model.geometryProfile}`,
          ] as const,
          [
            `Model build${timings.pipelineReused ? " (earlier run)" : ""}`,
            buildText(model.build),
          ] as const,
          ...(model.timing === undefined
            ? []
            : [["Model GPU time", timingText(model.timing)] as const]),
        ]),
  ];
};
