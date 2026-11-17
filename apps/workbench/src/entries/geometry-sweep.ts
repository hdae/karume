/**
 * The geometry sweep: measures the GEMM tile geometries on this host's GPU with `runGeometrySweep`
 * of `@karume/runtime/tune`, one call per case with idle time before every case but the first
 * (cooling: a case takes minutes, and heat moves the default geometry's re-measurement, which the
 * derivation checks), and keeps the per-case records merged into one `karume-geometry-sweep/3`
 * record — the input `deriveGeometryProfile` reads, with no envelope of its own. The evaluation set
 * `k88` (perf-ledger K-88) gets the K-88 summary; no table is derived from its record.
 * The run concludes a verdict from its rows (`sweepVerdict` of `../geometry-sweep/summary.ts`): a
 * candidate that differs from its default or errored is a `fail`.
 *
 * `runGeometrySweep` acquires and destroys a dedicated device per call, so this entry acquires none.
 */

import { geometrySweepCases } from "@karume/runtime/tune";

import type { Entry } from "../entry.ts";
import { foldSweepEvent, INITIAL_LIVE, runSweepPlan } from "../geometry-sweep/run.ts";
import {
  CANDIDATE_SETS,
  planSweep,
  SWEEP_CASE_IDS,
  SWEEP_OPS,
  sweepOps,
  sweepSelection,
} from "../geometry-sweep/settings.ts";
import { geometrySweepResult, progressLine } from "../geometry-sweep/summary.ts";

const entry: Entry = {
  id: "geometry-sweep",
  title: "Geometry sweep",
  description:
    "The GEMM tile geometries measured case by case on this GPU (runGeometrySweep, idle time between cases), merged into the one karume-geometry-sweep/3 record deriveGeometryProfile reads. The candidate set k88 is an evaluation (perf-ledger K-88): no table is derived from its record.",
  parameters: [
    {
      kind: "choice",
      name: "candidateSet",
      label: "Candidate set",
      options: CANDIDATE_SETS,
      default: "quick+",
    },
    {
      kind: "choices",
      name: "ops",
      label: "Op families",
      options: SWEEP_OPS,
      default: sweepOps(geometrySweepCases()),
    },
    {
      kind: "choices",
      name: "cases",
      label: "Cases (none: every case of the chosen op families in the chosen set)",
      options: SWEEP_CASE_IDS,
      default: [],
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Rounds per geometry (the min is kept)",
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
      default: 90,
    },
  ],
  run: async ({ parameters, gpuOptions, signal, progress }) => {
    const selection = sweepSelection(parameters);
    const plan = planSweep(selection, geometrySweepCases(selection.candidateSet));
    let live = INITIAL_LIVE;
    const show = (): void => progress(progressLine(plan, live, performance.now()));
    // The idle wait reports nothing for 90 s by default: a ticker counts it down.
    let ticker: ReturnType<typeof setInterval> | undefined;
    try {
      const outcome = await runSweepPlan(plan, {
        gpu: gpuOptions.gpu,
        signal,
        onEvent: (event) => {
          live = foldSweepEvent(live, event);
          if (live.phase.kind === "idle") {
            ticker ??= setInterval(show, 1000);
          } else {
            clearInterval(ticker);
            ticker = undefined;
          }
          show();
        },
      });
      return geometrySweepResult(plan, outcome);
    } finally {
      clearInterval(ticker);
    }
  },
};

export default entry;
