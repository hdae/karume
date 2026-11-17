/**
 * The GEMV sweep: measures the GEMV choices of `linear` on this host's GPU with `runGemvSweep` of
 * `@karume/runtime/tune` — per case and row count the baseline (what a Session at the level `exact`
 * runs with the default profile), the identical class (the sequential variants and the row-block
 * heights, whose words must be the baseline's) and the reorder class (the parallel GEMV at each lane
 * count, plain, staged and subgroup, held to the band of `linear` against an f64 oracle on the host),
 * each row count after a quick look that times in full only the references and the kernels near
 * enough to matter. One call per case with idle time before every case but the first, the per-case
 * records merged into one `karume-gemv-sweep/3` record — what `deriveLinearGemvChoices` reads.
 *
 * The run concludes a verdict from its rows (`gemvSweepVerdict` of `../gemv-sweep/summary.ts`): an
 * identical candidate whose words differ, a reorder candidate outside the band, or a row that errored
 * is a `fail`. `runGemvSweep` acquires and destroys a dedicated device per call, so this entry
 * acquires none.
 */

import { QUICK_LOOK_MARGIN } from "@karume/runtime/tune";

import type { Entry } from "../entry.ts";
import { foldGemvEvent, INITIAL_GEMV_LIVE, runGemvPlan } from "../gemv-sweep/run.ts";
import {
  GEMV_CANDIDATE_NAMES,
  GEMV_CANDIDATE_SETS,
  GEMV_CASE_IDS,
  GEMV_ROW_COUNTS,
  gemvSweepSelection,
  planGemvSweep,
} from "../gemv-sweep/settings.ts";
import { gemvProgressLine, gemvSweepResult } from "../gemv-sweep/summary.ts";
import { liveLine } from "../per-case-sweep.ts";

const entry: Entry = {
  id: "gemv-sweep",
  title: "GEMV sweep",
  description:
    "The GEMV choices of linear (sequential variant, row-block height, parallel lanes, staged, subgroup) measured case by case on this GPU (runGemvSweep, idle time between cases, a quick look before the timing), merged into one karume-gemv-sweep/3 record: identical candidates must write the baseline's words, reordering ones must stay in the band of linear.",
  parameters: [
    {
      kind: "choice",
      name: "candidateSet",
      label:
        "Candidate set (quick: M = 1; full: M = 1, 4 and 8, the whole grid; row-blocks: M = 16 and 64)",
      options: GEMV_CANDIDATE_SETS,
      default: "quick",
    },
    {
      kind: "choices",
      name: "cases",
      label: "Cases (none: every case)",
      options: GEMV_CASE_IDS,
      default: [],
    },
    {
      kind: "choices",
      name: "rows",
      label: "Row counts (none: every row count of the set)",
      options: GEMV_ROW_COUNTS,
      default: [],
    },
    {
      kind: "choices",
      name: "candidates",
      label: "Candidates besides the references (none: every candidate)",
      options: GEMV_CANDIDATE_NAMES,
      default: [],
    },
    {
      kind: "number",
      name: "quickLookMargin",
      label: "Quick look margin (0: off — every kernel is timed in full)",
      min: 0,
      max: 1000,
      default: QUICK_LOOK_MARGIN,
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Rounds per kernel (the min is kept)",
      min: 1,
      max: Number.MAX_SAFE_INTEGER,
      default: 5,
    },
    {
      kind: "integer",
      name: "idleSeconds",
      label: "Idle before every case but the first (s)",
      min: 0,
      max: Number.MAX_SAFE_INTEGER,
      default: 10,
    },
  ],
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
