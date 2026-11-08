// expert_linear without a GPU (ADR 0145 decisions 2 and 3): the CPU reference against a hand
// computation, the plan-time producer rule of its index, the implemented GPU subset, the support
// check and the residency of its weight slot. The shape rule is pinned by the conformance table
// (fixtures/op-contracts.json, both sides); the GPU paths by gpu_expert_linear_test.ts and
// gpu_expert_linear_session_test.ts.
//
// Expected values and byte counts are written out by hand (deriving them from the code under test
// would make the checks tautological).

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { RuntimeSupportError } from "../src/ops/support.ts";
import { OpContractError } from "../src/ops.ts";
import {
  applyReferenceOp,
  decodeExpertLinearWeight,
  ReferenceOpError,
  refTensor,
} from "../src/reference/ops.ts";
import { ExecutionError, validateGraphContracts } from "../src/runtime/plan.ts";
import { assertWeightsWithinLimits, planWeightResidency } from "../src/runtime/weight-residency.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import { autoTensors, mergeGraph, type StorageSpec } from "./helpers/merged-graph.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

const f32 = (shape: readonly number[], values: readonly number[]) =>
  refTensor(shape, Float32Array.from(values));
const i32 = (shape: readonly number[], values: readonly number[]) =>
  refTensor(shape, Int32Array.from(values));

// Two experts [2, out 2, in 3].
const WEIGHT = f32([2, 2, 3], [1, 2, 3, 0, -1, 1, 2, 0, -1, 1, 1, 1]);
const INDEX = i32([2, 2], [1, 0, 0, 1]);

describe("the CPU reference of expert_linear", () => {
  it("multiplies the shared x row of each token by the picked expert's rows", () => {
    // t0 = [1, 1, 1] picks e1, e0; t1 = [2, -1, 0.5] picks e0, e1.
    const x = f32([2, 3], [1, 1, 1, 2, -1, 0.5]);
    const y = applyReferenceOp("expert_linear", [x, INDEX, WEIGHT]);
    assertEquals(y.shape, [2, 2, 2]);
    assertEquals([...y.data], [1, 3, 6, 0, 1.5, 1.5, 3.5, 1.5]);
  });

  it("reads one x row per pick in the per-pick form", () => {
    const x = f32([2, 2, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]);
    const y = applyReferenceOp("expert_linear", [x, INDEX, WEIGHT]);
    assertEquals([...y.data], [2, 1, 2, -1, 3, 1, 1, 3]);
  });

  it("throws on an id outside [0, E) instead of reading another row (ADR 0061)", () => {
    const x = f32([2, 3], [1, 1, 1, 2, -1, 0.5]);
    for (const bad of [-1, 2]) {
      assertThrows(
        () => applyReferenceOp("expert_linear", [x, i32([2, 2], [1, 0, bad, 1]), WEIGHT]),
        ReferenceOpError,
        `expert_linear index[1, 0] = ${bad} is out of range of the 2 experts`,
      );
    }
  });

  it("refuses an i32 x and an f32 index (the per-slot contract f32 / i32 / f32)", () => {
    assertThrows(
      () => applyReferenceOp("expert_linear", [i32([2, 3], [1, 1, 1, 2, 1, 0]), INDEX, WEIGHT]),
      OpContractError,
    );
    assertThrows(
      () =>
        applyReferenceOp("expert_linear", [
          f32([2, 3], [1, 1, 1, 2, -1, 0.5]),
          f32([2, 2], [1, 0, 0, 1]),
          WEIGHT,
        ]),
      OpContractError,
    );
  });

  it("decodes i4 with f32 steps and with f16 steps to the same values (rows = the leading axis)", () => {
    // [2, 1, 16] with group 16: expert 0 holds q = 1 (u 9), expert 1 q = -2 (u 6); steps 0.5 and
    // 0.25 (f16 0x3800 / 0x3400).
    const bytes = Uint8Array.from([...new Array(8).fill(0x99), ...new Array(8).fill(0x66)]);
    const expected = [...new Array(16).fill(0.5), ...new Array(16).fill(-0.5)];
    const i4 = decodeExpertLinearWeight([2, 1, 16], {
      storage: "i4",
      bytes,
      steps: Float32Array.from([0.5, 0.25]),
      groupSize: 16,
    });
    const i4f16 = decodeExpertLinearWeight([2, 1, 16], {
      storage: "i4f16",
      bytes,
      steps: Uint8Array.from([0x00, 0x38, 0x00, 0x34]),
      groupSize: 16,
    });
    assertEquals([...i4.data], expected);
    assertEquals([...i4f16.data], expected);
  });
});

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

const E = 4;
const K = 2;
const OUT = 3;
const IN = 32;

type IndexSource = "topk" | "input" | "initializer" | "reshape" | "topk-over-5";

/**
 * `y = expert_linear(x, index, w)` with `w [E, out, in]`, the index produced by `source`:
 * `topk(scores [T, E])` (the accepted form), a graph input, an i32 initializer, a reshape of a
 * topk's indices, or a topk over 5 values instead of E.
 */
const expertGraph = (
  source: IndexSource = "topk",
  options: { readonly out?: number; readonly inFeatures?: number; readonly extra?: string } = {},
): DeclarationJson => {
  const { out = OUT, inFeatures = IN, extra } = options;
  const tokens: number | string = source === "initializer" ? 2 : "T";
  const scoresWidth = source === "topk-over-5" ? 5 : E;
  const nodes: DeclarationJson["nodes"] = [];
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype: "f32", shape: [tokens, inFeatures] },
  ];
  const values: DeclarationJson["values"] = {
    w: { dtype: "f32", shape: [E, out, inFeatures] },
    y: { dtype: "f32", shape: [tokens, K, out] },
  };
  const initializers: DeclarationJson["initializers"] = { w: {} };
  const ops = new Set(["expert_linear"]);
  if (source === "input") {
    inputs.push({ name: "index", dtype: "i32", shape: [tokens, K] });
  } else if (source === "initializer") {
    initializers["index"] = {};
    values["index"] = { dtype: "i32", shape: [tokens, K] };
  } else {
    inputs.push({ name: "scores", dtype: "f32", shape: [tokens, scoresWidth] });
    const picked = source === "reshape" ? "picked" : "index";
    values["top"] = { dtype: "f32", shape: [tokens, K] };
    values[picked] = { dtype: "i32", shape: [tokens, K] };
    nodes.push({ op: "topk", ins: ["scores"], outs: ["top", picked], attrs: { k: K } });
    ops.add("topk");
    if (source === "reshape") {
      values["index"] = { dtype: "i32", shape: [tokens, K] };
      nodes.push({ op: "reshape", ins: ["picked"], outs: ["index"], attrs: {} });
      ops.add("reshape");
    }
  }
  nodes.push({ op: "expert_linear", ins: ["x", "index", "w"], outs: ["y"], attrs: {} });
  const outputs = ["y"];
  if (extra === "add") {
    // w also read outside any weight slot.
    inputs.push({ name: "z", dtype: "f32", shape: [E, out, inFeatures] });
    values["s"] = { dtype: "f32", shape: [E, out, inFeatures] };
    nodes.push({ op: "add", ins: ["z", "w"], outs: ["s"], attrs: {} });
    outputs.push("s");
    ops.add("add");
  } else if (extra === "linear") {
    // w also read at linear's weight slot (the planner does not look at shapes — the shape rule
    // would refuse this graph later; the residency rule under test comes first).
    inputs.push({ name: "v", dtype: "f32", shape: [2, inFeatures] });
    initializers["b"] = {};
    values["b"] = { dtype: "f32", shape: [E] };
    values["l"] = { dtype: "f32", shape: [2, E] };
    nodes.push({ op: "linear", ins: ["v", "w", "b"], outs: ["l"], attrs: {} });
    outputs.push("l");
    ops.add("linear");
  } else if (extra === "output") {
    outputs.push("w");
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...ops].sort() },
    symbols: tokens === "T" ? ["T"] : [],
    inputs,
    outputs,
    initializers,
    values,
    nodes,
  };
};

const I4: StorageSpec = { codec: "int4-sym-g", groupSize: 32 };
const I4F16: StorageSpec = { codec: "int4-sym-g-f16", groupSize: 32 };

// ---------------------------------------------------------------------------
// The plan-time producer rule (ADR 0145 decision 2)
// ---------------------------------------------------------------------------

describe("the producer rule of an expert_linear index (at Session construction)", () => {
  it("accepts output 1 of a topk over the weight's E", () => {
    validateGraphContracts(mergeGraph(expertGraph("topk")));
  });

  const refused: readonly [IndexSource, string][] = [
    ["input", "index 'index' comes from a graph input"],
    ["initializer", "index 'index' comes from an initializer"],
    ["reshape", "index 'index' is output 0 of reshape"],
    ["topk-over-5", "index 'index' comes from a topk over 5 values"],
  ];
  for (const [source, message] of refused) {
    it(`refuses an index from ${source === "topk-over-5" ? "a topk over another extent" : `a ${source}`}`, () => {
      assertThrows(
        () => validateGraphContracts(mergeGraph(expertGraph(source))),
        ExecutionError,
        `${message}; an expert_linear index MUST be output 1 of a topk over the weight's E (4) — ADR 0145 decision 2`,
      );
    });
  }
});

describe("the implemented GPU subset of expert_linear (at Session construction)", () => {
  it("refuses an i4 input width that is not a multiple of the 32-element word", () => {
    assertThrows(
      () =>
        validateGraphContracts(
          mergeGraph(expertGraph("topk", { inFeatures: 48 }), {
            w: { codec: "int4-sym-g", groupSize: 16 },
          }),
        ),
      ExecutionError,
      "input width 48 of weight 'w' (i4, group 16) cannot run on the GPU",
    );
  });

  it("refuses an input width that the group does not divide", () => {
    // out 4 · in 32 = 128 divides by 64 (the merge layer accepts it), in 32 does not.
    assertThrows(
      () =>
        validateGraphContracts(
          mergeGraph(expertGraph("topk", { out: 4 }), {
            w: { codec: "int4-sym-g-f16", groupSize: 64 },
          }),
        ),
      ExecutionError,
      "input width 32 of weight 'w' (i4f16, group 64) cannot run on the GPU",
    );
  });

  it("refuses an f32 input width that is not a multiple of 4", () => {
    assertThrows(
      () => validateGraphContracts(mergeGraph(expertGraph("topk", { inFeatures: 6 }))),
      ExecutionError,
      "input width 6 of weight 'w' (f32) cannot run on the GPU",
    );
  });
});

// ---------------------------------------------------------------------------
// The support check and the residency of the weight slot
// ---------------------------------------------------------------------------

describe("the support check of int4-sym-g-f16", () => {
  it("admits it where expert_linear's weight slot is its only reader", () => {
    const graph = expertGraph("topk");
    openGraph(graph, autoTensors(graph, { w: I4F16 }));
  });

  it("refuses it when another op reads it too", () => {
    const graph = expertGraph("topk", { extra: "add" });
    assertThrows(
      () => openGraph(graph, autoTensors(graph, { w: I4F16 })),
      RuntimeSupportError,
      "unsupported storage 'i4f16' (1): w",
    );
  });
});

describe("the residency of an expert_linear weight", () => {
  // [4, 3, 32]: 384 codes = 192 bytes; 12 groups of 32 = 12 steps (24 bytes as f16, 48 as f32).
  it("keeps int4-sym-g-f16 resident in its own class, the f16 steps padded to whole words", () => {
    assertEquals(planWeightResidency(mergeGraph(expertGraph(), { w: I4F16 })).get("w"), {
      seat: "i4f16",
      payloadBytes: 192,
      scaleBytes: 24,
      groupSize: 32,
    });
  });

  it("keeps int4-sym-g resident as i4 with f32 steps", () => {
    assertEquals(planWeightResidency(mergeGraph(expertGraph(), { w: I4 })).get("w"), {
      seat: "i4",
      payloadBytes: 192,
      scaleBytes: 48,
      groupSize: 32,
    });
  });

  it("keeps f32 as raw bytes", () => {
    assertEquals(planWeightResidency(mergeGraph(expertGraph())).get("w"), {
      seat: "raw",
      payloadBytes: 1536,
    });
  });

  it("refuses int4-sym-g-f16 that linear reads too (the f16 steps are expert_linear's alone)", () => {
    assertThrows(
      () => planWeightResidency(mergeGraph(expertGraph("topk", { extra: "linear" }), { w: I4F16 })),
      ExecutionError,
      "initializer 'w': int4-sym-g-f16 (f16 steps) is read by expert_linear only — ADR 0145",
    );
  });

  // int2-off / ternary never get here for [E, out, in]: the merge layer takes them at rank 2 only.
  for (const codec of ["int8-sym", "f16", "bf16"] as const) {
    it(`refuses ${codec} storage loudly`, () => {
      assertThrows(
        () => planWeightResidency(mergeGraph(expertGraph(), { w: codec })),
        ExecutionError,
        `initializer 'w': expert_linear reads f32 / int4-sym-g / int4-sym-g-f16 weights only, not '${codec}'`,
      );
    });
  }

  for (const [extra, why] of [
    ["add", "also read outside any weight slot"],
    ["output", "also a graph output"],
  ] as const) {
    it(`never falls back to the host f32 expansion for an i4 weight ${why}`, () => {
      assertThrows(
        () => planWeightResidency(mergeGraph(expertGraph("topk", { extra }), { w: I4 })),
        ExecutionError,
        "initializer 'w': an expert_linear weight is never expanded to f32 on the host",
      );
    });
  }

  it("compares the [E, out, in] payload and its step table with the device's binding limit", () => {
    // 192 B of codes over a 100 B limit; the 24 B step table fits.
    assertThrows(
      () =>
        assertWeightsWithinLimits(planWeightResidency(mergeGraph(expertGraph(), { w: I4F16 })), {
          maxStorageBufferBindingSize: 100,
          maxBufferSize: 1 << 20,
        }),
      ExecutionError,
      "1 weight buffers exceed the device limits (detected before allocation):\n" +
        "  - initializer 'w' payload (residency class i4f16, allocated 192 bytes): exceeds" +
        " maxStorageBufferBindingSize 100 bytes by 92 bytes",
    );
  });

  it("counts the codec's bytes as compressed in the estimate", () => {
    const graph = expertGraph();
    const { resident } = openGraph(graph, autoTensors(graph, { w: I4F16 })).estimate({
      bindings: { T: 5 },
    });
    assertEquals(resident.weights.compressedBytes, 192 + 24);
    assertEquals(resident.weights.expandedBytes, 0);
  });
});
