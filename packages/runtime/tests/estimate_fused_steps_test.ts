// The memory estimate counts the i8a8 linear's node temporaries: the a8 linear with and without a
// folded producer (helpers/fused-window-graphs.ts), its `workspaceBytes` pinned to a number derived
// here by hand — the output's `numel × 4`, the temporaries the node declares, and the placement's
// two rules that matter at this size (a value read and a value written by one dispatch go to
// different regions; two slots alive at once share a region only at an offset aligned to 256, which
// opens a new region instead when it would grow by the whole slot — transient-plan.ts). Graph
// inputs and weights have no slot. The literals are the independent side of the estimate ==
// execution check (gpu_estimate_test.ts compares the same graphs with a real Session's slot table).
// No GPU.

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { UNREAD_ATTENTION_DEVICE } from "../src/runtime/attention-route.ts";
import { estimateFusionFacts } from "../src/runtime/estimate.ts";
import { planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import { sessionFusionContext } from "../src/runtime/session-fusion-context.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import { a8LinearCase, type FusedWindowCase } from "./helpers/fused-window-graphs.ts";

/** Desktop-class limits (no binding limit decides anything at these sizes). */
const LIMITS = {
  maxStorageBufferBindingSize: 2 ** 31 - 4,
  maxBufferSize: 2 ** 32,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

/** The plan's counter of the case's rule (the same planning as a Session given the options). */
const windows = (fused: FusedWindowCase): number => {
  const model = openGraph(fused.graph, fused.tensors);
  const { graph } = model;
  const { maxComputeWorkgroupsPerDimension, ...facts } = estimateFusionFacts(
    planWeightResidency(graph),
    undefined,
    { ...LIMITS, ...fused.options },
    UNREAD_ATTENTION_DEVICE,
  );
  const { counts } = planFusions(
    planGraph(graph, fused.bindings ?? {}, undefined).nodes,
    sessionFusionContext({
      ...facts,
      useCounts: countUses(graph),
      outputNames: new Set(graph.outputs),
      limits: {
        maxStorageBufferBindingSize: LIMITS.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension,
      },
    }),
  );
  const record: Readonly<Record<string, number>> = counts;
  return record[fused.counter] ?? 0;
};

describe("the estimate counts the i8a8 linear's two node temporaries", () => {
  // x [1, 4, 64] → linear with w [16, 64] resident as i8 under linearCompute "a8": ① quantize reads x
  // (an input) and writes xq (4 rows · 64 / 4 words · 4 B = 256 B) and xs (4 rows · 4 B = 16 B);
  // ② the GEMM reads xq, w, b, xs and writes y [1, 4, 16] = 256 B. y is alive throughout and conflicts
  // with xq and xs (② reads them, writes y): y region 0, xq region 1; xs would sit at offset 256 in
  // region 1 (xq is alive) and grow it by its whole 16 B, so it opens region 2: 256 + 256 + 16.
  const a8 = 256 + 256 + 16;
  // y alone, the plain linear's one dispatch.
  const f32 = 256;
  for (const prologue of [false, true]) {
    const linear = a8LinearCase(prologue);
    it(`${linear.name}: xq and xs with the output; the folded producer's output is not counted`, () => {
      assertEquals(windows(linear), linear.windows, "linear prologue folded");
      const model = openGraph(linear.graph, linear.tensors);
      assertEquals(
        model.estimate({ ...LIMITS, ...linear.options }).scenarios[0].workspaceBytes,
        a8,
      );
      if (!prologue) {
        // Without the seat the linear is the plain one (the route needs linearCompute "a8").
        assertEquals(model.estimate({ ...LIMITS }).scenarios[0].workspaceBytes, f32);
      }
    });
  }
});
