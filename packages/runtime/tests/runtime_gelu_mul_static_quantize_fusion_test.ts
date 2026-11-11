/**
 * The `geluMulStaticQuantize` fusion rule without a GPU (perf-ledger K-48 step 1b): the accepted
 * set (the QAT per-layer-input gate) and every rejection. Bit identity is the job of
 * `gpu_gelu_mul_static_quantize_fusion_test.ts`.
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { prologueUnaryWgsl } from "../src/codegen/elementwise.ts";
import { mergedGraph } from "../src/format/container/bind.ts";
import {
  GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE,
  geluMulStaticQuantizeKey,
  type GeluMulOrder,
  geluMulStaticQuantizeWgsl,
} from "../src/kernels/gelu-mul-static-quantize.ts";
import { STATIC_QUANTIZE_FN_WGSL, staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { GELU_MUL_STATIC_QUANTIZE_RULE } from "../src/runtime/fusion-rules/gelu-mul-static-quantize.ts";
import {
  FUSION_RULES,
  type FusionPlan,
  fusionMatchContext,
  planFusions,
} from "../src/runtime/fusion.ts";
import { countUses, type NodePlan, planGraph } from "../src/runtime/plan.ts";
import { sessionOptionsViolation } from "../src/runtime/session-options-check.ts";
import {
  geluMulStaticQuantizeGraph,
  type GeluMulStaticQuantizeGraphOptions,
} from "./helpers/gelu-mul-static-quantize-graph.ts";
import { GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

type FusionContext = Parameters<typeof planFusions>[1];

const planned = (options: GeluMulStaticQuantizeGraphOptions = {}) => {
  const graph = mergedGraph(
    memoryModel(geluMulStaticQuantizeGraph(options), []).graphs[GRAPH_NAME],
    GRAPH_NAME,
  );
  const context: FusionContext = {
    rules: FUSION_RULES,
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
  };
  return { nodes: planGraph(graph, {}).nodes, context };
};

/** Plans with `fuseGeluMulStaticQuantize: true`, or the option given (`"omit"` leaves it unset). */
const fuse = (
  options: GeluMulStaticQuantizeGraphOptions = {},
  enabled: boolean | "omit" = true,
): FusionPlan => {
  const { nodes, context } = planned(options);
  return planFusions(nodes, {
    ...context,
    ...(enabled === "omit" ? {} : { fuseGeluMulStaticQuantize: enabled }),
  });
};

const outline = (plan: FusionPlan): readonly string[] =>
  plan.steps.map((step) => (step.kind === "fused" ? `fused:${step.rule}` : step.plan.node.op));

const ORDERS: readonly GeluMulOrder[] = ["gelu-other", "other-gelu"];

describe("geluMulStaticQuantize: the accepted chain", () => {
  it("folds gelu_tanh → mul → static_quantize into one dispatch for both operand orders, both operand forms and 1 to 8 rows", () => {
    for (const rows of [1, 4, 8]) {
      for (const order of ORDERS) {
        for (const other of ["slice", "input"] as const) {
          const label = `rows ${rows} / ${order} / ${other}`;
          const scale = Math.fround(0.0371);
          const plan = fuse({ rows, order, other, scale });
          assertEquals(
            outline(plan),
            [
              ...(other === "slice" ? ["slice", "reshape"] : []),
              "fused:geluMulStaticQuantize",
              "neg",
            ],
            label,
          );
          assertEquals(plan.counts.geluMulStaticQuantize, 1, label);
          const step = plan.steps.find((s) => s.kind === "fused");
          if (step?.kind !== "fused") throw new Error(`${label}: not folded`);
          // Bind order: gate, then the mul's other operand (whatever its slot), then the output.
          assertEquals(step.binds, ["gate", "ple"], label);
          assertEquals(step.ins, ["gate", "ple"], label);
          assertEquals(step.nodeCount, 3, label);
          assertEquals(step.outputName, "y", label);
          assertEquals(step.outputShape, [1, rows, 256], label);
          const [dispatch] = step.dispatches;
          const n = rows * 256;
          assertEquals(dispatch.key, geluMulStaticQuantizeKey(order), label);
          assertEquals(dispatch.workgroups, {
            kind: "gridStride",
            items: n,
            size: GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE,
          });
          // n, then the two rounding masks at 0 (run-time words), then the standalone SRQ table.
          assertEquals([...dispatch.params.slice(0, 4)], [n, 0, 0, 0], label);
          assertEquals(dispatch.params.slice(4), staticQuantizeParams(n, scale), label);
          if (other === "slice") {
            // The QAT operand is the reshape alias of the materialised slice: the fold binds that
            // alias, whose buffer is the slice's in the element order the plain mul reads.
            const reshape = plan.steps[1];
            assert(reshape.kind === "node" && reshape.aliasesInput, `${label}: reshape alias`);
          }
        }
      }
    }
  });

  it("folds scale 0 (the identity SRQ) with the identity flag in the table", () => {
    const step = fuse({ scale: 0 }).steps.find((s) => s.kind === "fused");
    if (step?.kind !== "fused") throw new Error("not folded");
    assertEquals(step.dispatches[0].params[4 + 258], 1);
  });

  it("keeps the plain chain when the option is omitted or false", () => {
    for (const enabled of ["omit", false] as const) {
      const plan = fuse({}, enabled);
      assertEquals(plan.counts.geluMulStaticQuantize, 0);
      assertEquals(outline(plan), [
        "slice",
        "reshape",
        "gelu_tanh",
        "mul",
        "static_quantize",
        "neg",
      ]);
    }
  });
});

describe("geluMulStaticQuantize: forms that stay on the plain chain", () => {
  it("rejects an extra consumer, a graph output, a broadcast or duplicate operand, a non-adjacent chain and unmeasured shapes", () => {
    const cases: readonly (readonly [string, GeluMulStaticQuantizeGraphOptions])[] = [
      ["gelu value with another consumer", { extraGeluConsumer: true }],
      ["product with another consumer", { extraMulConsumer: true }],
      ["gelu value as a graph output", { geluOutput: true }],
      ["product as a graph output", { mulOutput: true }],
      ["broadcast operand [1, 1, 256]", { broadcast: true }],
      ["broadcast plain input [1, 1, 256]", { broadcast: true, other: "input" }],
      ["the other operand is the gate", { otherIsGate: true }],
      ["a node between gelu_tanh and mul", { interpose: true }],
      ["9 rows", { rows: 9 }],
      ["64 rows", { rows: 64 }],
      ["last dim 128", { dim: 128 }],
      ["last dim 512", { dim: 512 }],
    ];
    for (const [label, options] of cases) {
      const plan = fuse(options);
      assertEquals(plan.counts.geluMulStaticQuantize, 0, label);
      assert(
        plan.steps.every((step) => step.kind === "node"),
        `${label}: ${outline(plan).join(" ")}`,
      );
    }
  });

  it("rejects a chain with a non-f32 slot", () => {
    const { nodes, context } = planned();
    // A planned list whose mul reads an i32 operand (the contract admits mul on i32; the chain
    // cannot be built from IR with gelu_tanh, so the slot is rewritten after planning).
    const rewritten = nodes.map((node): NodePlan =>
      node.node.op === "mul" ? { ...node, inputDtypes: ["f32", "i32"] } : node,
    );
    const plan = planFusions(rewritten, { ...context, fuseGeluMulStaticQuantize: true });
    assertEquals(plan.counts.geluMulStaticQuantize, 0);
  });

  it("does not fold an SRQ whose output is handed over as packed int8 (ADR 0105)", () => {
    const { nodes, context } = planned({ other: "input" });
    const scan = {
      ...context,
      fuseGeluMulStaticQuantize: true,
      matchContext: fusionMatchContext(context),
    };
    assert(
      GELU_MUL_STATIC_QUANTIZE_RULE.apply(nodes, 0, { ...scan, packedValues: new Map() }) !==
        undefined,
      "the same window folds when the SRQ is not packed",
    );
    assertEquals(
      GELU_MUL_STATIC_QUANTIZE_RULE.apply(nodes, 0, {
        ...scan,
        packedValues: new Map([["y", Math.fround(0.0371)]]),
      }),
      undefined,
    );
  });
});

describe("geluMulStaticQuantize: the kernel text", () => {
  it("compiles the product's own gelu_tanh, mul and SRQ text, in the node's operand order", () => {
    for (const order of ORDERS) {
      const wgsl = geluMulStaticQuantizeWgsl(order);
      const gelu = prologueUnaryWgsl("gelu_tanh", "v0");
      assert(wgsl.includes(`bitcast<u32>(${gelu.value}) ^ params.mask_act`), order);
      for (const helper of gelu.helpers) assert(wgsl.includes(helper), `${order}: helper`);
      assert(wgsl.includes(STATIC_QUANTIZE_FN_WGSL), `${order}: SRQ`);
      const product = order === "gelu-other" ? "act * v1" : "v1 * act";
      assert(wgsl.includes(`quantize(bitcast<u32>(${product}) ^ params.mask_mul)`), order);
    }
    assert(geluMulStaticQuantizeKey("gelu-other") !== geluMulStaticQuantizeKey("other-gelu"));
  });
});

describe("fuseGeluMulStaticQuantize option", () => {
  it("accepts a boolean on its own and rejects anything else", () => {
    for (const value of [true, false]) {
      assertEquals(sessionOptionsViolation({ fuseGeluMulStaticQuantize: value }), undefined);
    }
    for (const value of [null, 0, "true"]) {
      const options = {};
      Reflect.set(options, "fuseGeluMulStaticQuantize", value);
      assertEquals(
        sessionOptionsViolation(options),
        "options.fuseGeluMulStaticQuantize must be a boolean",
      );
    }
  });
});
