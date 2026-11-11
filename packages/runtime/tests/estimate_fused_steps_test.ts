// The memory estimate counts every fused step in its post-fusion form (P4 — ADR 0148 for the
// external rules) and the i8a8 linear's node temporaries: one graph per rule with exactly one
// window, graph inputs in and the window's output out, and the a8 linear with and without a folded
// producer (helpers/fused-window-graphs.ts); each one's `workspaceBytes` pinned to a number derived
// here by hand — the output's `numel × 4`, the temporaries the rule declares, and the placement's
// two rules that matter at this size (a value read and a value written by one dispatch go to
// different regions; two slots alive at once share a region only at an offset aligned to 256, which
// opens a new region instead when it would grow by the whole slot — transient-plan.ts). Graph
// inputs and weights have no slot. Nothing here goes through fused-step-layout.ts: the literals are
// the independent side of the estimate == execution check (gpu_estimate_test.ts compares the same
// graphs with a real Session's slot table). No GPU.

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { UNREAD_ATTENTION_DEVICE } from "../src/runtime/attention-route.ts";
import { estimateFusionFacts } from "../src/runtime/estimate.ts";
import { planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import { sessionFusionContext } from "../src/runtime/session-fusion-context.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import {
  a8LinearCase,
  FUSED_WINDOW_CASES,
  type FusedWindowCase,
} from "./helpers/fused-window-graphs.ts";

/** Desktop-class limits (no binding limit decides anything at these sizes). */
const LIMITS = {
  maxStorageBufferBindingSize: 2 ** 31 - 4,
  maxBufferSize: 2 ** 32,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

/**
 * The hand-derived workspace of each case (bytes). `y`-like outputs are graph outputs: pinned,
 * alive to the end.
 */
const EXPECTED: Readonly<Record<string, number>> = {
  // y [4, 256] f32 = 4,096 B. The step reads x (an input), w and its scale (resident) and b.
  linearStaticQuantize: 4 * 256 * 4,
  // y [1, 4, 256] = 4,096 B; gelu_tanh's and mul's outputs are folded away.
  geluMulStaticQuantize: 4 * 256 * 4,
  // y [4, 256] = 4,096 B; the normalised value is folded away.
  rmsNormAdd: 4 * 256 * 4,
  // y [4, 8] = 128 B; sigmoid's output is folded away.
  silu: 4 * 8 * 4,
  // y [2, 3, 10, 14] = 840 elements = 3,360 B; the expands' copies are folded away.
  upsample2x: 2 * 3 * 10 * 14 * 4,
  // y [1, 1, 5, 128] = 2,560 B; the halves, the rotated copy and both products are folded away.
  rope: 5 * 128 * 4,
  // q_y [1, 4, 2, 128] = 4,096 B.
  rmsNormRope: 4 * 256 * 4,
  // q_y, k_y and v_out [1, 4, 2, 128] = 4,096 B each, all graph outputs: alive together, and a
  // second slot in a region would grow it by its whole size, so three regions. v_out is neg of v_x's
  // reshape (an alias of an input — no slot). k's normalised value and reshapes, which q's window
  // defers, are folded into k's window.
  "rmsNormRope (Wan form, deferred nodes)": 3 * 4 * 256 * 4,
  // y [1, 4, 8] = 128 B; the two passthrough reshapes alias graph inputs.
  adaln: 4 * 8 * 4,
  // y 128 B + the pack of the four modulation vectors 4 · 8 · 4 = 128 B: dispatch ① writes the
  // pack, ② reads it and writes y, so they take two regions.
  "adaln (per-token form)": 4 * 8 * 4 + 4 * 8 * 4,
  // out [1, 4, 8] = 128 B.
  gatedResidual: 4 * 8 * 4,
  // One row block (2 heads · 64 rows · 64 keys · 4 = 32,768 B per temporary: S, mask-added S, P)
  // and y [2, 64, 8] = 4,096 B. S / mask-added S are read / written by one dispatch (two regions);
  // P reuses S's bytes (S is released before P exists) but not the region mask-added S is in; y,
  // alive throughout, conflicts with P (④ reads P, writes y), and in mask-added S's region it would
  // sit at offset 32,768 and grow it by its whole size, so it opens a third region:
  // 32,768 · 2 + 4,096.
  rowBlockAttention: 2 * 32768 + 4096,
  // y [1, 4, 64] = 1,024 B; gelu's output is folded away.
  "external test/gelu_mul": 4 * 64 * 4,
  // y 1,024 B + the staged gelu value 1,024 B (① writes it, ② reads it and writes y: two regions).
  // The unfused pair has the same total by construction (the temporary takes the gelu output's
  // place); the GPU equality and the fault injection are what pin the temporary's mapping.
  "external test/gelu_mul_staged": 4 * 64 * 4 + 4 * 64 * 4,
};

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

describe("the estimate counts a fused step in its post-fusion form", () => {
  for (const make of FUSED_WINDOW_CASES) {
    const fused = make();
    it(`${fused.name}: the window's output and its rule's temporaries, nothing it folds away`, () => {
      // Precondition: the window is taken (a literal of an unfused plan would check nothing here).
      assertEquals(windows(fused), fused.windows, "windows taken");
      const report = openGraph(fused.graph, fused.tensors).estimate({
        ...LIMITS,
        ...(fused.bindings === undefined ? {} : { bindings: fused.bindings }),
        ...fused.options,
      });
      assertEquals(report.scenarios[0].workspaceBytes, EXPECTED[fused.name]);
    });
  }
});
