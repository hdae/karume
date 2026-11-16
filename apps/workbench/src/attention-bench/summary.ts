/**
 * What a person reads of an attention bench run: the status line while it runs, and after it the
 * device facts the routes were chosen from, one line per shape and route, and with the candidate one
 * line per shape of its ratios and identity. Pure, over the runtime's bench record.
 */

import type { AttentionBenchProgress, AttentionBenchReport } from "@karume/runtime/tune";

import {
  benchTableRows,
  type BenchTableRow,
  candidateRows,
  type CandidateTableRow,
  formatMs,
  formatRatio,
} from "./rows.ts";

/** The live status: the shapes done so far and the runtime's latest message or finished shape. */
export const benchProgressText = (
  progress: AttentionBenchProgress,
  done: number,
  total: number,
): string =>
  `${done} / ${total} shapes · ${progress.kind === "status" ? progress.message : `${progress.result.shape.id} finished`}`;

const joined = (parts: readonly (string | undefined)[]): string =>
  parts.filter((part) => part !== undefined).join(" · ");

/** Timing unit, the integer dot-product variant (and its canary) and the subgroup facts. */
const statusLine = (report: AttentionBenchReport): string => {
  const { dp4a, device } = report.variants;
  const canary =
    dp4a.canary === undefined
      ? ""
      : ` (canary ${dp4a.canary.branch}${dp4a.canary.exact ? "" : ", inexact"})`;
  return joined([
    report.timing.unit === "ns"
      ? "timing: GPU timestamps (ns)"
      : `timing: ${report.timing.unit} clock`,
    `integer dot: ${dp4a.dot === "dp4a" ? "dp4a" : "emulated"}${dp4a.listed ? "" : " (WGSL feature not listed)"}${canary}`,
    `subgroups ${String(device.subgroups)} (features ${String(device.features.subgroups)} / size control ${String(device.features.subgroupSizeControl)} / subgroup_id ${String(device.features.subgroupId)} / lanes linear ${String(device.subgroupLanesLinear)})`,
    `geometry profile ${report.variants.geometryProfile}`,
  ]);
};

const routeLine = (row: BenchTableRow): string => {
  const head = `${row.shape} ${row.route}: `;
  const note = row.note === undefined ? "" : ` — ${row.note}`;
  if (row.route === "quantization") {
    return `${head}${joined([`node ${formatMs(row.nodeMs)} ms`, row.stages])}${note}`;
  }
  // A route production would not run here: the row holds only the reason.
  if (row.rows === undefined) return `${head}${row.note ?? ""}`;
  return `${head}${joined([
    `${formatMs(row.msPerDispatch)} ms / dispatch`,
    `${formatMs(row.msPer1000Rows)} ms / 1,000 rows`,
    `node ${formatMs(row.nodeMs)} ms`,
    row.ratio === undefined ? undefined : `online / s16 ×${row.ratio.toFixed(3)}`,
    row.split,
    row.variant,
  ])}${note}`;
};

const candidateLine = (row: CandidateTableRow): string => {
  const head = `${row.shape} tiled ÷ online: `;
  if (row.identical === undefined) return `${head}${row.note ?? ""}`;
  return `${head}${joined([
    `kernel ${formatRatio(row.kernelRatio)}`,
    `kernel + producer ${formatRatio(row.withProducerRatio)}`,
    `${row.rounds ?? 0} rounds`,
    row.identical ? "identical" : "differs",
  ])}${row.note === undefined ? "" : ` — ${row.note}`}`;
};

/**
 * The summary of a bench record: the status facts, then per shape and route the ms per dispatch, the
 * ms per 1,000 query rows, the node estimate and on the online row the online / s16 ratio (below 1 =
 * online faster), then with the candidate per shape tiled ÷ online for the kernel alone and with its
 * producer (below 1 = the candidate faster) and whether the outputs are identical.
 */
export const attentionBenchSummary = (report: AttentionBenchReport): string[] => [
  statusLine(report),
  ...benchTableRows(report.shapes).map(routeLine),
  ...candidateRows(report.shapes).map(candidateLine),
];
