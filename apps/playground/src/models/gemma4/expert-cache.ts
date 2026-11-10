/**
 * The Gemma 4 chat page's status row for the exact expert cache (`Pipeline.expertCacheReport()` —
 * ADR 0146): the counters over the pipeline's lifetime and the per-run and per-expert figures they
 * give. Pure, so `vp test` covers it.
 */

import type { ExpertCacheReport } from "@karume/models/gemma";

import { type ExpertCacheSlots, FITTED_EXPERT_SLOTS, sameExpertSlots } from "./settings";

const MIB = 2 ** 20;
const GIB = 2 ** 30;

/** The figures derived from a report; a ratio over zero is `undefined` (nothing to divide). */
export type ExpertCacheFigures = {
  /** Per routed layer, in layer order (`ExpertCacheReport.slots`). */
  readonly slots: readonly number[];
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
  /** Present exactly when the pipeline was built with a self-speculation depth. */
  readonly speculation?: ExpertSpeculationFigures;
};

/** The self-speculation cycles' figures (`ExpertCacheReport.speculation`). */
export type ExpertSpeculationFigures = {
  readonly cycles: number;
  /** Drafts confirmed per cycle (at most the depth). */
  readonly acceptedDraftsPerCycle?: number;
  /** Cycles that ended at the verify's first missing row. */
  readonly partialCommits: number;
  /** Verify runs issued again because their first row missed. */
  readonly verifyRetries: number;
};

const speculationFigures = (
  speculation: NonNullable<ExpertCacheReport["speculation"]>,
): ExpertSpeculationFigures => ({
  cycles: speculation.cycles,
  ...(speculation.cycles === 0
    ? {}
    : { acceptedDraftsPerCycle: speculation.acceptedDrafts / speculation.cycles }),
  partialCommits: speculation.partialCommits,
  verifyRetries: speculation.verifyRetries,
});

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
  ...(report.speculation === undefined
    ? {}
    : { speculation: speculationFigures(report.speculation) }),
});

/**
 * The slots as the status names them: the count when every layer has it, else the mean and the
 * range, after "fitted" when they are {@link FITTED_EXPERT_SLOTS}.
 */
export const expertSlotsText = (slots: ExpertCacheSlots): string => {
  if (typeof slots === "number") return `${slots} slots`;
  const fewest = Math.min(...slots);
  const most = Math.max(...slots);
  if (fewest === most) return `${fewest} slots`;
  const mean = slots.reduce((sum, count) => sum + count, 0) / slots.length;
  const perLayer = `${Number.isInteger(mean) ? mean : mean.toFixed(1)} slots per layer on average (${fewest}–${most})`;
  return sameExpertSlots(slots, FITTED_EXPERT_SLOTS) ? `fitted, ${perLayer}` : perLayer;
};

/** One line for the status card. */
export const expertCacheText = (report: ExpertCacheReport): string => {
  const figures = expertCacheFigures(report);
  const retries = figures.retriesPerCommittedRun;
  const perExpert = figures.loadMsPerExpert;
  const { speculation } = figures;
  const accepted = speculation?.acceptedDraftsPerCycle;
  return [
    expertSlotsText(figures.slots),
    `${figures.committedRuns} committed runs`,
    `${retries === undefined ? "—" : retries.toFixed(2)} retries / committed run`,
    ...(speculation === undefined
      ? []
      : [
          `${speculation.cycles} cycles`,
          `${accepted === undefined ? "—" : accepted.toFixed(2)} accepted drafts / cycle`,
          `${speculation.partialCommits} partial commits`,
          `${speculation.verifyRetries} verify retries`,
        ]),
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

/** The expert cache of a load request (`expertCache`), as the status's Pipeline line names it. */
export const expertCacheRequestText = (expertCache: {
  readonly slots: ExpertCacheSlots;
  readonly hostBytes?: number;
  readonly depth?: number;
}): string =>
  [
    `expert cache ${expertSlotsText(expertCache.slots)}`,
    ...(expertCache.hostBytes === undefined ? [] : [expertHostTierText(expertCache.hostBytes)]),
    ...(expertCache.depth === undefined ? [] : [`depth ${expertCache.depth}`]),
  ].join(" · ");
