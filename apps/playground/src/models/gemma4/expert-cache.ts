/**
 * The Gemma 4 chat page's status row for the exact expert cache (`Pipeline.expertCacheReport()` —
 * ADR 0146): the counters over the pipeline's lifetime and the per-run and per-expert figures they
 * give. Pure, so `vp test` covers it.
 */

import type { ExpertCacheReport } from "@karume/models/gemma";

const MIB = 2 ** 20;
const GIB = 2 ** 30;

/** The figures derived from a report; a ratio over zero is `undefined` (nothing to divide). */
export type ExpertCacheFigures = {
  readonly slots: number;
  readonly committedRuns: number;
  /** Re-runs a miss cost, per committed run. */
  readonly retriesPerCommittedRun?: number;
  readonly loadedExperts: number;
  /** Of {@link loadedExperts}, the ones uploaded from the host tier (RAM). */
  readonly hostExperts: number;
  /** Of {@link loadedExperts}, the ones read from the distribution. */
  readonly containerExperts: number;
  readonly evictedExperts: number;
  /** Read from the distribution. */
  readonly readMiB: number;
  readonly loadMs: number;
  /** Load wall time per loaded expert. */
  readonly loadMsPerExpert?: number;
  /** Of {@link loadMs}, the wait for the distribution's reads. */
  readonly readMs: number;
  /** Of {@link loadMs}, the staging ring's uploads. */
  readonly uploadMs: number;
  /** What the host tier holds now (0 without one). */
  readonly hostHeldExperts: number;
  readonly hostHeldGiB: number;
};

export const expertCacheFigures = (report: ExpertCacheReport): ExpertCacheFigures => ({
  slots: report.slots,
  committedRuns: report.committedRuns,
  ...(report.committedRuns === 0
    ? {}
    : { retriesPerCommittedRun: report.retries / report.committedRuns }),
  loadedExperts: report.loadedExperts,
  hostExperts: report.hostExperts,
  containerExperts: report.containerExperts,
  evictedExperts: report.evictedExperts,
  readMiB: report.bytesRead / MIB,
  loadMs: report.loadMs,
  ...(report.loadedExperts === 0 ? {} : { loadMsPerExpert: report.loadMs / report.loadedExperts }),
  readMs: report.readMs,
  uploadMs: report.uploadMs,
  hostHeldExperts: report.hostHeldExperts,
  hostHeldGiB: report.hostHeldBytes / GIB,
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
    `${figures.loadedExperts} loaded (${figures.hostExperts} from the host tier, ${figures.containerExperts} from the distribution)`,
    `${figures.evictedExperts} evicted`,
    `${figures.readMiB.toFixed(1)} MiB read`,
    `load ${figures.loadMs.toFixed(0)} ms (${perExpert === undefined ? "—" : perExpert.toFixed(2)} ms / expert; read ${figures.readMs.toFixed(0)} ms, upload ${figures.uploadMs.toFixed(0)} ms)`,
    `host tier holds ${figures.hostHeldExperts} experts (${figures.hostHeldGiB.toFixed(2)} GiB)`,
  ].join(" · ");
};

/** The host tier of a load request (`expertCache.hostBytes`), as the status names it. */
export const expertHostTierText = (hostBytes: number): string =>
  hostBytes === Number.MAX_SAFE_INTEGER
    ? "host tier all"
    : `host tier ${(hostBytes / GIB).toFixed(2)} GiB`;
