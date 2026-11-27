/**
 * The GEMV sweep: measures the GEMV choices of `linear` on this host's GPU with `runGemvSweep` of
 * `@karume/runtime/tune` — per case and row count the baseline (what a Session at the level `exact`
 * runs with the default profile), the identical class (the sequential variants and the row-block
 * heights, whose words must be the baseline's) and the reorder class (the parallel GEMV at each lane
 * count, plain, staged and subgroup, held to the band of `linear` against an f64 oracle on the host),
 * each row count after a quick look that times in full only the references and the kernels near
 * enough to matter; at M = 4 and 8 the reorder candidates that the case's M = 1 group of the same run
 * leaves unadoptable are skipped. One call per case with idle time before every case but the first,
 * the per-case records merged into one `karume-gemv-sweep/4` record — what
 * `deriveLinearGemvChoices` reads. The `explore` set's sentinels go first, and each later call takes
 * the sentinels the earlier calls measured, so the merged record is what one call over the cases
 * writes.
 *
 * The run concludes a verdict from its rows (`gemvSweepVerdict` of
 * `@karume/sweep-app-support/gemv-sweep/summary`): an identical candidate whose words differ, a
 * reorder candidate outside the band, or a row that errored is a `fail`. `runGemvSweep` acquires and
 * destroys a dedicated device per call, so this entry acquires none. The parameters, the plan, the
 * run and the record are the shared package's, which the benchmark app uses too.
 */

import {
  foldGemvEvent,
  INITIAL_GEMV_LIVE,
  runGemvPlan,
} from "@karume/sweep-app-support/gemv-sweep/run";
import {
  GEMV_SWEEP_ENTRY_ID,
  GEMV_SWEEP_PARAMETERS,
  gemvSweepSelection,
  planGemvSweep,
} from "@karume/sweep-app-support/gemv-sweep/settings";
import { gemvProgressLine, gemvSweepResult } from "@karume/sweep-app-support/gemv-sweep/summary";
import { liveLine } from "@karume/sweep-app-support/per-case-sweep";

import type { Entry } from "../entry.ts";

const entry: Entry = {
  id: GEMV_SWEEP_ENTRY_ID,
  title: "GEMV sweep",
  description:
    "The GEMV choices of linear (sequential variant, row-block height, parallel lanes, staged, subgroup) measured case by case on this GPU (runGemvSweep, idle time between cases, a quick look before the timing, M = 4 and 8 only for the reorder candidates M = 1 leaves adoptable; the explore set measures the whole grid on a few sentinel cases first and the candidates they predict on the others), merged into one karume-gemv-sweep/4 record: identical candidates must write the baseline's words, reordering ones must stay in the band of linear.",
  parameters: GEMV_SWEEP_PARAMETERS,
  run: async ({ parameters, gpuOptions, signal, progress }) => {
    const plan = planGemvSweep(gemvSweepSelection(parameters));
    let live = INITIAL_GEMV_LIVE;
    const line = liveLine(() => progress(gemvProgressLine(plan, live, performance.now())));
    try {
      const outcome = await runGemvPlan(plan, {
        ...(gpuOptions.gpu === undefined ? {} : { gpu: gpuOptions.gpu }),
        signal,
        onEvent: (event) => {
          live = foldGemvEvent(live, event);
          line.update(live.phase.kind === "idle");
        },
      });
      return gemvSweepResult(plan, outcome);
    } finally {
      line.stop();
    }
  },
};

export default entry;
