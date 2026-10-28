/**
 * What the Status card says while a run loads the pipeline and estimates, including which stages are
 * not cancellable. Pure, so `vp test` covers it.
 */

import type { AssetProgress } from "@karume/hub";

import type { RunProgress } from "@/hooks/use-run-task";

const MIB = 2 ** 20;

/** Before the first progress event: the manifest and the admission checks (Cancel stops it). */
export const LOAD_START: RunProgress = { stage: "loading the pipeline" };

/**
 * The note while the Session is built. `fromPretrained`'s signal reaches only the fetch layer: the
 * Session build (reading the fetched parts, creating the buffers, uploading the weights) takes none.
 */
export const SESSION_BUILD_NOTE =
  "The Session build takes no signal: Cancel is unavailable until it ends.";

/** The note while the estimate runs: `estimate()` takes no signal. */
export const ESTIMATE_NOTE = "estimate() takes no signal: Cancel is unavailable until it returns.";

/** The stage of the estimate itself (not cancellable). */
export const ESTIMATING: RunProgress = {
  stage: "estimating",
  cancellable: false,
  cancelNote: ESTIMATE_NOTE,
};

/**
 * The stage of one load progress event. The hub reports one stream for the whole model, with
 * `total` the size of every file it fetches, so `loaded` reaching `total` means every file is in and
 * what follows inside `fromPretrained` is the Session build.
 */
export const loadProgress = (progress: AssetProgress): RunProgress =>
  progress.loaded >= progress.total
    ? { stage: "building the Session", cancellable: false, cancelNote: SESSION_BUILD_NOTE }
    : {
        stage: "fetching the weights",
        count: {
          done: Math.round(progress.loaded / MIB),
          total: Math.round(progress.total / MIB),
          unit: "MiB",
        },
        note: progress.path,
      };
