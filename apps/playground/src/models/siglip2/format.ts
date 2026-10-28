/**
 * The record as the result card shows it: the summary rows and the embeddings table (one row per
 * statistic, one column per image). Pure, so `vp test` covers it.
 */

import type { AssetProgress } from "@karume/hub";

import type { RunProgress } from "@/hooks/use-run-task";
import { type BuildCost, type ComponentTiming, seconds, topOpShares } from "@/lib/record";

import type { EmbeddingRecord, Siglip2Record } from "./record";
import type { ImageSlot } from "./request";

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
 * The status line of a build from its latest progress event. `loaded` / `total` count every file of
 * the load and `fileLoaded` / `fileTotal` the file of `path`, so the note keeps the two apart. Once
 * `loaded` reaches `total` every file is fetched and `fromPretrained` goes on to build the Session
 * (reading the cached weight parts and uploading them to the GPU). That part takes no AbortSignal
 * and streams no progress, so it is its own stage, not cancellable — rather than leaving the last
 * fetch event on screen as if nothing were happening.
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

export const slotLabel = (slot: ImageSlot): string => (slot === "a" ? "Image A" : "Image B");

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

/** The cosine row: the value, or why there is none (absent for a one-image run). */
export const comparisonText = (record: Siglip2Record): string | undefined => {
  const comparison = record.output.comparison;
  if (comparison === undefined) return undefined;
  return comparison.cosineSimilarity === null
    ? `— (${comparison.reason})`
    : comparison.cosineSimilarity.toFixed(6);
};

/** The summary rows of the result card (the JSON keeps the full detail). */
export const summaryRows = (record: Siglip2Record): readonly (readonly [string, string])[] => {
  const { timings, diagnostics } = record;
  const vision = diagnostics.vision;
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
    [
      "Build s",
      timings.buildSeconds === null
        ? "— (pipeline reused from an earlier run)"
        : String(timings.buildSeconds),
    ],
    ...record.request.images.map(
      ({ slot }, index) =>
        [`Embed ${slotLabel(slot)} s`, String(timings.embedSeconds[index] ?? "—")] as const,
    ),
    ["Wall s", String(timings.wallSeconds)],
    ...(vision === undefined
      ? []
      : [
          [
            "Vision Session",
            `${vision.runs} runs · weights ${vision.weightsAllocatedGiB} GiB · ${vision.submitCount} submits · ${vision.dispatchCount} dispatches (this run; Session total ${vision.sessionSubmitCount} / ${vision.sessionDispatchCount}) · geometry ${vision.geometryProfile}`,
          ] as const,
          [
            `Vision build${timings.pipelineReused ? " (earlier run)" : ""}`,
            buildText(vision.build),
          ] as const,
          ...(vision.timing === undefined
            ? []
            : [["Vision GPU time", timingText(vision.timing)] as const]),
        ]),
  ];
};

/** One row of the embeddings table: a statistic and its value per image, in request order. */
export type EmbeddingRow = { readonly label: string; readonly values: readonly string[] };

const summaryValue = (
  embedding: EmbeddingRecord,
  key: "l2Norm" | "min" | "max" | "mean",
): string => (embedding.summary === undefined ? "—" : embedding.summary[key].toFixed(6));

export const embeddingRows = (record: Siglip2Record): readonly EmbeddingRow[] => {
  const images = record.request.images;
  const embeddings = record.output.embeddings;
  const row = (label: string, value: (embedding: EmbeddingRecord) => string): EmbeddingRow => ({
    label,
    values: embeddings.map(value),
  });
  return [
    {
      label: "Input",
      values: images.map(({ name, width, height }) => `${name} · ${width}×${height}`),
    },
    row("Dimension", ({ dimension }) => String(dimension)),
    row("Non-finite", ({ nonFinite }) => String(nonFinite)),
    row("L2 norm", (embedding) => summaryValue(embedding, "l2Norm")),
    row("Min", (embedding) => summaryValue(embedding, "min")),
    row("Max", (embedding) => summaryValue(embedding, "max")),
    row("Mean", (embedding) => summaryValue(embedding, "mean")),
    row("First values", ({ first }) =>
      first.map((value) => (typeof value === "number" ? value.toFixed(4) : value)).join(", "),
    ),
    row("sha256 (f32 bytes)", ({ sha256 }) => sha256),
    { label: "Input sha256 (RGB8)", values: images.map(({ rgbSha256 }) => rgbSha256) },
  ];
};
