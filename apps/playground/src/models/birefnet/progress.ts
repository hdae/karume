/**
 * The load stages of `birefnet.Pipeline.fromPretrained` as the page reports them. The aggregated
 * progress counts the container parts once each: it ends at `loaded === total` when the parts are
 * in the browser's cache (Cached) or, on Direct, at once (a local source has nothing to warm, so
 * every part reports a single `complete`). The Session is built after that and reads every weight
 * block (from the cache, or from `/models/` on Direct) and uploads it. That build takes no signal,
 * so the page names it as its own, not cancellable stage. Pure, so `vp test` covers it.
 */

import type { RunProgress } from "@/hooks/use-run-task";

const MIB = 2 ** 20;

export const BUILD_NOTE =
  "Reads the weights (from the browser's cache, or from /models/ on Direct) and uploads them to the GPU. This step takes no signal: Cancel is unavailable until it ends.";

export const loadingProgress = (progress: {
  readonly loaded: number;
  readonly total: number;
  readonly path: string;
}): RunProgress => {
  const count = {
    done: Math.round(progress.loaded / MIB),
    total: Math.round(progress.total / MIB),
    unit: "MiB",
  };
  return progress.loaded < progress.total
    ? { stage: "loading weights", count, note: progress.path }
    : { stage: "building the Session", count, cancellable: false, cancelNote: BUILD_NOTE };
};

export const SEGMENT_NOTE = "Segmentation takes no signal: Cancel is unavailable until it ends.";
