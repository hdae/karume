// The memory estimate counts every fused step in its post-fusion form (P4 — ADR 0148 for the
// external rules) and the i8a8 linear's node temporaries: one graph per rule with exactly one
// window, graph inputs in and the window's output out, and the a8 linear with and without a folded
// producer (helpers/fused-window-graphs.ts); each one's `workspaceBytes` pinned to a number derived
// here by hand — the output's `numel × 4`, the temporaries the rule declares, and the placement's
// two rules that matter at this size (a value read and a value written by one dispatch go to
// different regions; two slots alive at once share a region only at an offset aligned to 256, which
// opens a new region instead when it would grow by the whole slot — transient-plan.ts). Graph
// inputs and weights have no slot. The expected literals do not go through fused-step-layout.ts (the
// estimate under test does): they are the independent side of the estimate == execution check (gpu_estimate_test.ts compares the same
// graphs with a real Session's slot table). The last section checks that a fused step execution
// refuses gets the same refusal from the estimate (execution on a fake GPU). No GPU.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { defineRule, type FusionMatch, type NodePlan } from "../kernel.ts";
import { PipelineKeyConflictError } from "../src/gpu/pipeline-cache.ts";
import { UNREAD_ATTENTION_DEVICE } from "../src/runtime/attention-route.ts";
import { estimateFusionFacts } from "../src/runtime/estimate.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { planFusions } from "../src/runtime/fusion.ts";
import { countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import { sessionFusionContext } from "../src/runtime/session-fusion-context.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import { fakeRecipeGpuContext } from "./helpers/fake-gpu.ts";
import {
  a8LinearAfterDeadRegion,
  a8LinearCase,
  FUSED_WINDOW_CASES,
  type FusedWindowCase,
  geluMulGraph,
} from "./helpers/fused-window-graphs.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

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
  // dead = neg(z) 8,192 B, released once probe = slice(dead) (4 B, a graph output) has read it;
  // then y 1,024 B and the staged value 1,024 B. Placed by size: dead opens region 0; y joins it at
  // offset 0 (dead is released before y exists); the staged value is written by ① and read by ②,
  // which writes y, so it may not join y's region and opens region 1 — were that read lost, it would
  // sit in region 0 at offset 1,024 for free; probe conflicts with dead (slice reads dead, writes
  // probe), and in region 1 it would grow it by its whole 4 B, so it opens region 2.
  "external test/gelu_mul_staged after a dead region": 8192 + 1024 + 4,
  // dead 8,192 B and probe 4 B as above; then gate = neg(x) 1,024 B (an intermediate, released when
  // the window's step has read it) and y 1,024 B. Placed by size: dead opens region 0; gate joins it
  // at offset 0 (dead is released before gate exists); the step's dispatch reads gate and writes y,
  // so y may not join gate's region and opens region 1 — were that read lost, y would sit in region
  // 0 at offset 1,024 for free (8,192 + 4 in all); probe conflicts with dead, and in region 1 it
  // would grow it by its whole 4 B (y is alive from its offset 0), so it opens region 2.
  "external test/gelu_mul binding an intermediate after a dead region": 8192 + 1024 + 4,
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
  it("a8 linear after a dead region: the GEMM's read of xq keeps xq out of y's region", () => {
    // dead = neg(z) 8,192 B, released once probe = slice(dead) (4 B, a graph output) has read it;
    // then y 256 B, xq 256 B, xs 16 B. Placed by size: dead opens region 0; y joins it at offset 0
    // (dead is released before y exists); xq is read by ②, which writes y, so it may not join y's
    // region and opens region 1 — were that read lost, it would sit in region 0 at offset 256 for
    // free; xs conflicts with y too and in region 1 (xq alive) would grow it by its whole 16 B, so it
    // opens region 2; probe conflicts with dead and opens region 3 (4 B) for the same reason.
    const linear = a8LinearAfterDeadRegion();
    const model = openGraph(linear.graph, linear.tensors);
    assertEquals(
      model.estimate({ ...LIMITS, ...linear.options }).scenarios[0].workspaceBytes,
      8192 + 256 + 16 + 4,
    );
  });
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

describe("the estimate refuses a fused step execution refuses", () => {
  type GeluMulMatch = FusionMatch & {
    readonly gate: string;
    readonly other: string;
    readonly count: number;
  };

  /** How {@link twoDispatchRule} departs from the staged rule's schedule and keys. */
  type Variant = {
    /** The dispatch after which temporary 0 is released (default 1, after ②). */
    readonly releaseAfter?: number;
    /** ②'s pipeline key (default its own, `ext:test:two_dispatch:mul`). */
    readonly mulKey?: string;
    /** Both kernels' text ends with a comment naming the element count (default not). */
    readonly countInText?: boolean;
  };

  /** A kernel's bind group: the params, then one storage binding per operand. */
  const kernel = (bindings: readonly ("read" | "read_write")[], tail = ""): string =>
    [
      "@group(0) @binding(0) var<uniform> params: vec4<u32>;",
      ...bindings.map(
        (access, slot) =>
          `@group(0) @binding(${slot + 1}) var<storage, ${access}> b${slot}: array<f32>;`,
      ),
      `@compute @workgroup_size(64) fn main() {}${tail}`,
    ].join("\n");

  /**
   * `mul(gelu(gate), other)` of {@link geluMulGraph} in two dispatches with the gelu value in
   * temporary 0: ① binds gate and the temporary, ② the temporary, other and the output — the
   * schedule of the staged rule (helpers/fused-window-graphs.ts), with `variant` applied.
   */
  const twoDispatchRule = ({ releaseAfter = 1, mulKey, countInText = false }: Variant = {}) =>
    defineRule<GeluMulMatch, "test/two_dispatch">({
      name: "test/two_dispatch",
      heads: ["gelu"],
      contract: "E",
      internals: "private",
      match: (nodes: readonly NodePlan[], index: number): GeluMulMatch | undefined => {
        const [gelu, mul] = [nodes[index], nodes[index + 1]];
        if (gelu?.node.op !== "gelu" || mul?.node.op !== "mul") return undefined;
        if (mul.node.ins[0] !== gelu.outputs[0].name) return undefined;
        return {
          window: [gelu, mul],
          chain: [gelu, mul],
          gate: gelu.node.ins[0],
          other: mul.node.ins[1],
          count: gelu.inputShapes[0].reduce((product, dim) => product * dim, 1),
        };
      },
      build: (matched) => {
        const product = matched.chain[1].outputs[0];
        const workgroups = { kind: "gridStride", items: matched.count, size: 64 } as const;
        const tail = countInText ? ` // n = ${matched.count}` : "";
        return {
          binds: [matched.gate, matched.other],
          outputName: product.name,
          outputShape: product.shape,
          temps: [{ byteLength: matched.count * 4, allocBefore: 0, releaseAfter }],
          dispatches: [
            {
              key: "ext:test:two_dispatch:gelu",
              wgsl: () => kernel(["read", "read_write"], tail),
              params: new Uint32Array(4),
              operands: [
                { kind: "bind", index: 0 },
                { kind: "temp", id: 0 },
              ],
              workgroups,
            },
            {
              key: mulKey ?? "ext:test:two_dispatch:mul",
              wgsl: () => kernel(["read", "read", "read_write"], tail),
              params: new Uint32Array(4),
              operands: [{ kind: "temp", id: 0 }, { kind: "bind", index: 1 }, { kind: "output" }],
              workgroups,
            },
          ],
        };
      },
    });

  /** The run of a Session given `rule` on a fake GPU (it stops at the transient plan at the latest). */
  const execute = async (rule: ReturnType<typeof twoDispatchRule>): Promise<unknown> => {
    const session = await createSessionFromContainer(
      fakeRecipeGpuContext(),
      await openModelBytes(geluMulGraph(), []),
      GRAPH_NAME,
      { fusionRules: [rule] },
    );
    const row = [1, 4, 64];
    return await session.run({ gate: fill(row, (i) => i), other: fill(row, (i) => -i) });
  };

  it("a temporary bound by a dispatch after its release: execution's declaration check refuses it, and the estimate with the same message", async () => {
    const rule = twoDispatchRule({ releaseAfter: 0 });
    const message =
      "step 'y': dispatch 1 'ext:test:two_dispatch:mul' binding 1 references temporary 0 outside its" +
      " live range [0, 0]";
    // The refusal execution gives (the recipe's declaration check, not wrapped by the rule's name).
    await assertRejects(() => execute(rule), ExecutionError, message);
    assertThrows(
      () => openGraph(geluMulGraph()).estimate({ ...LIMITS, fusionRules: [rule] }),
      ExecutionError,
      message,
    );
  });

  it("a temporary bound within its live range: the run passes the declaration check, and the estimate returns a number", async () => {
    const rule = twoDispatchRule();
    // The control: the fake GPU's run gets past every recipe and stops at the transient plan.
    await assertRejects(() => execute(rule), Error, "invalid transient plan limit");
    // y 1,024 B + the temporary 1,024 B (① writes it, ② reads it and writes y: two regions) — the
    // staged rule's case above.
    assertEquals(
      openGraph(geluMulGraph()).estimate({ ...LIMITS, fusionRules: [rule] }).scenarios[0]
        .workspaceBytes,
      4 * 64 * 4 + 4 * 64 * 4,
    );
  });

  /** The refusal of a second WGSL text under `key`, as execution reports it for the rule. */
  const keyConflict = (key: string): string =>
    "fusion rule 'test/two_dispatch': resolving its step's dispatches threw PipelineKeyConflictError:" +
    ` Pipeline key "${key}" was given a different WGSL (codegen determinism is broken)`;

  it("two dispatches of one step under one key with different WGSL: the pipeline cache refuses it, and the estimate with the same error", async () => {
    const rule = twoDispatchRule({ mulKey: "ext:test:two_dispatch:gelu" });
    const message = keyConflict("ext:test:two_dispatch:gelu");
    const executed = await assertRejects(() => execute(rule), ExecutionError, message);
    assert(executed.cause instanceof PipelineKeyConflictError, "execution keeps the cache's error");
    const estimated = assertThrows(
      () => openGraph(geluMulGraph()).estimate({ ...LIMITS, fusionRules: [rule] }),
      ExecutionError,
      message,
    );
    assert(estimated.cause instanceof PipelineKeyConflictError, "the estimate keeps it too");
  });

  it("one key whose WGSL differs between two scenarios of one estimate call: refused as the second shape's run would be", () => {
    // gelu·mul on [1, 1, M, 64] beside a state_append of gate, so generation binds the chunk rows M:
    // prefill M = chunkLength, decode M = 1.
    const row = [1, 1, "M", 64];
    const graph: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["state_append", "gelu", "mul"] },
      symbols: ["M", "C"],
      inputs: [
        { name: "gate", dtype: "f32", shape: row },
        { name: "other", dtype: "f32", shape: row },
      ],
      outputs: ["y"],
      initializers: {},
      values: { g: { dtype: "f32", shape: row }, y: { dtype: "f32", shape: row } },
      states: { k: { dtype: "f32", shape: [1, 1, "C", 64] } },
      nodes: [
        { op: "state_append", ins: ["gate"], outs: [], attrs: {}, states: { slot: "k" } },
        { op: "gelu", ins: ["gate"], outs: ["g"], attrs: {} },
        { op: "mul", ins: ["g", "other"], outs: ["y"], attrs: {} },
      ],
    };
    const estimate = (rule: ReturnType<typeof twoDispatchRule>, chunkLength: number) =>
      openGraph(graph).estimate({
        ...LIMITS,
        fusionRules: [rule],
        generation: { chunkLength, bindings: { C: 8 } },
      });
    const counted = twoDispatchRule({ countInText: true });
    // The controls: with one chunk row both scenarios have the same text, and a text that names no
    // shape gets a number for both scenarios of four rows.
    assertEquals(estimate(counted, 1).scenarios.length, 2);
    assertEquals(estimate(twoDispatchRule(), 4).scenarios.length, 2);
    // Prefill registers the 256-element text, decode brings the 64-element one under the same key.
    const estimated = assertThrows(
      () => estimate(counted, 4),
      ExecutionError,
      keyConflict("ext:test:two_dispatch:gelu"),
    );
    assert(estimated.cause instanceof PipelineKeyConflictError);
  });
});
