/**
 * The Gemma 4 chat page's status row for the exact expert cache (`Pipeline.expertCacheReport()` —
 * ADR 0146): the counters over the pipeline's lifetime and the per-run and per-expert figures they
 * give. Pure, so `vp test` covers it.
 */

import type { ExpertCacheReport } from "@karume/models/gemma";

const MIB = 2 ** 20;

/** The figures derived from a report; a ratio over zero is `undefined` (nothing to divide). */
export type ExpertCacheFigures = {
  readonly slots: number;
  readonly committedRuns: number;
  /** Re-runs a miss cost, per committed run. */
  readonly retriesPerCommittedRun?: number;
  readonly loadedExperts: number;
  readonly evictedExperts: number;
  readonly readMiB: number;
  readonly loadMs: number;
  /** Load wall time per loaded expert. */
  readonly loadMsPerExpert?: number;
};

export const expertCacheFigures = (report: ExpertCacheReport): ExpertCacheFigures => ({
  slots: report.slots,
  committedRuns: report.committedRuns,
  ...(report.committedRuns === 0
    ? {}
    : { retriesPerCommittedRun: report.retries / report.committedRuns }),
  loadedExperts: report.loadedExperts,
  evictedExperts: report.evictedExperts,
  readMiB: report.bytesRead / MIB,
  loadMs: report.loadMs,
  ...(report.loadedExperts === 0 ? {} : { loadMsPerExpert: report.loadMs / report.loadedExperts }),
});

/** One line for the status card. */
export const expertCacheText = (report: ExpertCacheReport): string => {
  const figures = expertCacheFigures(report);
  const retries = figures.retriesPerCommittedRun;
  const perExpert = figures.loadMsPerExpert;
  return [
    `${figures.slots} slots`,
    `${figures.committedRuns} committed runs`,
    `${retries === undefined ? "—" : retries.toFixed(2)} retries / committed run`,
    `${figures.loadedExperts} loaded`,
    `${figures.evictedExperts} evicted`,
    `${figures.readMiB.toFixed(1)} MiB read`,
    `load ${figures.loadMs.toFixed(0)} ms (${perExpert === undefined ? "—" : perExpert.toFixed(2)} ms / expert)`,
  ].join(" · ");
};
