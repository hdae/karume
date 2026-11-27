/**
 * The geometry sweep: measures the GEMM tile geometries on this host's GPU with `runGeometrySweep`
 * of `@karume/runtime/tune`, one call per case with idle time before every case but the first
 * (cooling: a case takes minutes, and heat moves the default geometry's re-measurement, which the
 * derivation checks), each case after a quick look that times in full only the default geometry and
 * the candidates within the margin of its time, and keeps the per-case records merged into one
 * `karume-geometry-sweep/4` record — the input `deriveGeometryProfile` reads, with no envelope of its own. The evaluation set
 * `k88` (perf-ledger K-88) gets the K-88 summary; no table is derived from its record. The
 * `explore` set measures `full`'s candidates class by class, each call given the rows of the calls
 * before it, so a candidate a case of its class ruled out is skipped in the later cases.
 * The run concludes a verdict from its rows (`sweepVerdict` of
 * `@karume/sweep-app-support/geometry-sweep/summary`): a candidate that differs from its default or
 * errored is a `fail`. The parameters, the plan, the run and the record are the shared package's,
 * which the benchmark app uses too.
 *
 * `runGeometrySweep` acquires and destroys a dedicated device per call, so this entry acquires none.
 */

import { geometrySweepCases } from "@karume/runtime/tune";
import {
  foldSweepEvent,
  INITIAL_LIVE,
  runSweepPlan,
} from "@karume/sweep-app-support/geometry-sweep/run";
import {
  GEOMETRY_SWEEP_ENTRY_ID,
  GEOMETRY_SWEEP_PARAMETERS,
  planSweep,
  sweepSelection,
} from "@karume/sweep-app-support/geometry-sweep/settings";
import {
  geometrySweepResult,
  progressLine,
} from "@karume/sweep-app-support/geometry-sweep/summary";
import { liveLine } from "@karume/sweep-app-support/per-case-sweep";

import type { Entry } from "../entry.ts";

const entry: Entry = {
  id: GEOMETRY_SWEEP_ENTRY_ID,
  title: "Geometry sweep",
  description:
    "The GEMM tile geometries measured case by case on this GPU (runGeometrySweep, idle time between cases, a quick look before the timing), merged into the one karume-geometry-sweep/4 record deriveGeometryProfile reads. The candidate set explore measures full's candidates class by class and skips, in the later cases of a class, a candidate an earlier one ruled out. The candidate set k88 is an evaluation (perf-ledger K-88): no table is derived from its record.",
  parameters: GEOMETRY_SWEEP_PARAMETERS,
  run: async ({ parameters, gpuOptions, signal, progress }) => {
    const selection = sweepSelection(parameters);
    const plan = planSweep(selection, geometrySweepCases(selection.candidateSet));
    let live = INITIAL_LIVE;
    // The idle wait reports nothing for 10 s by default: the live line counts it down.
    const line = liveLine(() => progress(progressLine(plan, live, performance.now())));
    try {
      const outcome = await runSweepPlan(plan, {
        gpu: gpuOptions.gpu,
        signal,
        onEvent: (event) => {
          live = foldSweepEvent(live, event);
          line.update(live.phase.kind === "idle");
        },
      });
      return geometrySweepResult(plan, outcome);
    } finally {
      line.stop();
    }
  },
};

export default entry;
