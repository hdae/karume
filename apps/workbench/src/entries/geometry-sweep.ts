/**
 * The geometry sweep: measures the GEMM tile geometries on this host's GPU with `runGeometrySweep`
 * of `@karume/runtime/tune`, one call per case with idle time before every case but the first
 * (cooling: a case takes minutes, and heat moves the default geometry's re-measurement, which the
 * derivation checks), each case after a quick look that times in full only the default geometry and
 * the candidates within the margin of its time, and keeps the per-case records merged into one
 * `karume-geometry-sweep/4` record — the input `deriveGeometryProfile` reads, with no envelope of its own. The evaluation set
 * `k88` (perf-ledger K-88) gets the K-88 summary; no table is derived from its record.
 * The run concludes a verdict from its rows (`sweepVerdict` of `../geometry-sweep/summary.ts`): a
 * candidate that differs from its default or errored is a `fail`.
 *
 * `runGeometrySweep` acquires and destroys a dedicated device per call, so this entry acquires none.
 */

import { geometrySweepCases, QUICK_LOOK_MARGIN } from "@karume/runtime/tune";

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
import { liveLine } from "../per-case-sweep.ts";

const entry: Entry = {
  id: "geometry-sweep",
  title: "Geometry sweep",
  description:
    "The GEMM tile geometries measured case by case on this GPU (runGeometrySweep, idle time between cases, a quick look before the timing), merged into the one karume-geometry-sweep/4 record deriveGeometryProfile reads. The candidate set k88 is an evaluation (perf-ledger K-88): no table is derived from its record.",
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
      kind: "text",
      name: "geometries",
      label:
        "Candidate geometries, comma-separated names (none: every candidate of the set; the default geometry is always measured)",
      default: "",
    },
    {
      kind: "number",
      name: "quickLookMargin",
      label: "Quick look margin (0: off — every geometry is timed in full)",
      min: 0,
      max: 1000,
      default: QUICK_LOOK_MARGIN,
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
    // The idle wait reports nothing for 90 s by default: the live line counts it down.
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
