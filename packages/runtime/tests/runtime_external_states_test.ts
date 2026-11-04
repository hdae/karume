// The check of the **declarations and contracts** of borrowed state slots (`states[].external`),
// readonly attention and shared initializers (ADR 0096 step 2 §1.1–§1.3). Uses no GPU at all.
//
// The borrower's (drafter's) graph is made of three new declarations:
//   ① `states[].external: true` — the instance lives in the lender context, not allocated here
//   ② attention with `attrs.readonly: true` — ins is q alone; it reads columns `[P−min(P,W), P)`
//   ③ `initializers[].shared` — no bytes in the distribution form; binds the lender's weight as is
//
// The three support one another (external slots are read only by readonly / readonly reads only
// external / no mixing), so **every form that writes only one side failing loudly** is the core of
// this check. If such a form got through, none would raise an exception; each would show up as **a
// different value**: "reading an uninitialized past" or "overwriting KV the lender has committed".

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { IrError, type IrGraph } from "../src/format/ir.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError, planGraph, validateGraphContracts } from "../src/runtime/plan.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

const HEADS = 4;
const KV_HEADS = 1;
const DEPTH = 8;
const CAPACITY = 16;
const WINDOW = 8;

/**
 * The minimal drafter: a graph where one readonly attention reads two borrowed slots.
 * The input is q `[1,H,1,D]` alone, and there is no `state_append` at all.
 */
const drafterGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention"] },
  symbols: [],
  inputs: [{ name: "q", dtype: "f32", shape: [1, HEADS, 1, DEPTH] }],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, 1, DEPTH] } },
  states: {
    "l13.k": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH], external: true },
    "l13.v": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH], external: true },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q"],
      outs: ["o"],
      attrs: { scale: 0.5, window: WINDOW, readonly: true },
      states: { k: "l13.k", v: "l13.v" },
    },
  ],
});

/** The minimal lender (own slots + a normal states-form attention + 2 appends). */
const targetGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: [],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, HEADS, 1, DEPTH] },
    { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
    { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, HEADS, 1, DEPTH] } },
  states: {
    "l13.k": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH] },
    "l13.v": { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5, window: WINDOW },
      states: { k: "l13.k", v: "l13.v" },
    },
    {
      op: "state_append",
      ins: ["k"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "l13.k" },
    },
    {
      op: "state_append",
      ins: ["v"],
      outs: [],
      attrs: { window: WINDOW },
      states: { slot: "l13.v" },
    },
  ],
});

/**
 * Declaration → merged graph (helpers/merged-graph.ts — the supply is generated from the
 * declaration).
 */
const parse = mergeGraph;

/**
 * Passes a mutated graph through parse + contract checks (each test states which layer it fails
 * in).
 */
const mutated = (mutate: (graph: DeclarationJson) => void): DeclarationJson => {
  const graph = drafterGraph();
  mutate(graph);
  return graph;
};

const slotShapes = (graph: IrGraph): ReadonlyMap<string, readonly number[]> =>
  new Map(Object.entries(graph.states).map(([name, slot]) => [name, slot.shape.map(Number)]));

test("the minimal drafter (2 external + 1 readonly attention) is accepted", () => {
  const graph = parse(drafterGraph());
  assertEquals(graph.states["l13.k"].external, true);
  validateGraphContracts(graph);
  const plan = planGraph(graph, {}, slotShapes(graph));
  // The output has the same shape as q ([B,H,1,D]).
  assertEquals(plan.nodes[0].outputs[0].shape, [1, HEADS, 1, DEPTH]);
  // ins is q alone (this step's k / v are not taken).
  assertEquals(plan.nodes[0].inputShapes.length, 1);
});

test("the lender (own slots + append) is not affected by a single byte (no external means no change)", () => {
  const graph = parse(targetGraph());
  assertEquals(graph.states["l13.k"].external, false);
  validateGraphContracts(graph);
});

test("state_append to an external slot is rejected (only the lender can write)", () => {
  const graph = parse(
    mutated((g) => {
      g.requires.ops = ["attention", "state_append"];
      g.inputs.push({ name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] });
      g.nodes.push({
        op: "state_append",
        ins: ["k"],
        outs: [],
        attrs: { window: WINDOW },
        states: { slot: "l13.k" },
      });
    }),
  );
  assertThrows(() => validateGraphContracts(graph), ExecutionError, "external, yet state_append");
});

test("an attention that is not readonly reading an external slot is rejected", () => {
  // The reader is put back into the normal states form (3 ins) = the form that "adds this step's
  // k/v". Allowing this for a slot with no writer always reads uninitialized rows as the past.
  const graph = parse(
    mutated((g) => {
      g.inputs.push(
        { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
        { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
      );
      g.nodes[0].ins = ["q", "k", "v"];
      g.nodes[0].attrs = { scale: 0.5, window: WINDOW };
    }),
  );
  assertThrows(
    () => validateGraphContracts(graph),
    ExecutionError,
    "is read by the non-readonly attention",
  );
});

test("a readonly reader reading a non-external slot is rejected", () => {
  const graph = parse(
    mutated((g) => {
      if (g.states === undefined) throw new Error("no states");
      delete g.states["l13.k"].external;
      delete g.states["l13.v"].external;
    }),
  );
  assertThrows(() => validateGraphContracts(graph), ExecutionError, "the non-external state slot");
});

test("mixing external and non-external is rejected (the borrower has no slots of its own)", () => {
  const graph = parse(
    mutated((g) => {
      if (g.states === undefined) throw new Error("no states");
      delete g.states["l13.v"].external;
    }),
  );
  // It fails on "readonly reads a non-external slot" before the mixing check (both the same layer
  // and type).
  assertThrows(() => validateGraphContracts(graph), ExecutionError, "ADR 0096 stage 2 §1.1");
});

test("a slot declared external with neither a readonly reader nor an append fails the reference check", () => {
  // A slot with zero readers is rejected by the IR layer (referential integrity) — external is no
  // exception.
  const graph = mutated((g) => {
    if (g.states === undefined) throw new Error("no states");
    g.states["l14.k"] = { dtype: "f32", shape: [1, KV_HEADS, CAPACITY, DEPTH], external: true };
  });
  assertThrows(() => parse(graph), IrError, "not referenced by any node");
});

test("states[].external cannot be anything but true (no two spellings meaning absent)", () => {
  for (const value of [false, 1, "true"] as const) {
    const graph = mutated((g) => {
      if (g.states === undefined) throw new Error("no states");
      // deno-lint-ignore no-explicit-any -- injects a value outside the contract and the types
      (g.states["l13.k"] as any).external = value;
    });
    assertThrows(() => parse(graph), IrError, "external");
  }
});

test("a readonly attention takes exactly one input (3 are rejected; the states field is required)", () => {
  const threeIns = parse(
    mutated((g) => {
      g.inputs.push(
        { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
        { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
      );
      g.nodes[0].ins = ["q", "k", "v"];
    }),
  );
  assertThrows(() => validateGraphContracts(threeIns), OpContractError, "takes exactly 1 inputs");

  const noStates = parse(
    mutated((g) => {
      g.inputs.push(
        { name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
        { name: "v", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] },
      );
      g.nodes[0].ins = ["q", "k", "v"];
      delete g.nodes[0].states;
      g.nodes[0].attrs = { scale: 0.5, readonly: true };
      delete g.states;
    }),
  );
  assertThrows(
    () => validateGraphContracts(noStates),
    OpContractError,
    "only on a node with a states field",
  );
});

test("attrs.readonly is true only and cannot be written on state_append", () => {
  const falsy = parse(
    mutated((g) => {
      g.nodes[0].attrs = { scale: 0.5, window: WINDOW, readonly: false };
    }),
  );
  assertThrows(() => validateGraphContracts(falsy), OpContractError, "readonly accepts only true");

  const onAppend = parse(
    mutated((g) => {
      g.requires.ops = ["attention", "state_append"];
      g.inputs.push({ name: "k", dtype: "f32", shape: [1, KV_HEADS, 1, DEPTH] });
      g.nodes.push({
        op: "state_append",
        ins: ["k"],
        outs: [],
        attrs: { window: WINDOW, readonly: true },
        states: { slot: "l13.k" },
      });
    }),
  );
  assertThrows(() => validateGraphContracts(onAppend), OpContractError, "has no attrs.readonly");
});

test("readonly shape checks: M = 1 MUST, B / D match, H % Hkv, window ≤ C", () => {
  const cases: readonly (readonly [string, (graph: DeclarationJson) => void, string])[] = [
    [
      "M is 2",
      (g) => {
        g.inputs[0].shape = [1, HEADS, 2, DEPTH];
        g.values.o.shape = [1, HEADS, 2, DEPTH];
      },
      "M (axis 2) of exactly 1",
    ],
    [
      "D does not match the slot",
      (g) => {
        g.inputs[0].shape = [1, HEADS, 1, DEPTH * 2];
        g.values.o.shape = [1, HEADS, 1, DEPTH * 2];
      },
      "B / D mismatch",
    ],
    [
      "H is not a multiple of Hkv",
      (g) => {
        g.inputs[0].shape = [1, HEADS + 1, 1, DEPTH];
        g.values.o.shape = [1, HEADS + 1, 1, DEPTH];
        if (g.states === undefined) throw new Error("no states");
        g.states["l13.k"].shape = [1, 2, CAPACITY, DEPTH];
        g.states["l13.v"].shape = [1, 2, CAPACITY, DEPTH];
      },
      "is not a positive integer multiple",
    ],
    [
      "window exceeds the capacity",
      (g) => {
        g.nodes[0].attrs = { scale: 0.5, window: CAPACITY + 1, readonly: true };
      },
      "slot capacity",
    ],
    [
      "k / v slots differ in shape",
      (g) => {
        if (g.states === undefined) throw new Error("no states");
        g.states["l13.v"].shape = [1, KV_HEADS, CAPACITY * 2, DEPTH];
      },
      "differ in shape",
    ],
  ];
  for (const [name, mutate, message] of cases) {
    const graph = parse(mutated(mutate));
    assertThrows(
      () => planGraph(graph, {}, slotShapes(graph)),
      OpContractError,
      message,
      `${name}: not rejected`,
    );
  }
});

/**
 * The lender's initializer name (§1.3 — the borrower's declared name MUST be **the same as the
 * lender's**).
 */
const SHARED_NAME = "model.lm_head.weight";

/**
 * A drafter with one shared initializer (§1.3) (the form that borrows the embedding weight from the
 * lender).
 */
const sharedGraph = (): DeclarationJson => {
  const graph = drafterGraph();
  graph.requires.ops = ["attention", "embedding"];
  graph.inputs.push({ name: "token", dtype: "i32", shape: [1] });
  graph.outputs.push("embed");
  graph.initializers[SHARED_NAME] = { shared: true };
  graph.values[SHARED_NAME] = { dtype: "f32", shape: [16, DEPTH] };
  graph.values["embed"] = { dtype: "f32", shape: [1, DEPTH] };
  graph.nodes.push({
    op: "embedding",
    ins: [SHARED_NAME, "token"],
    outs: ["embed"],
    attrs: { padding_idx: -1 },
  });
  return graph;
};

test("a shared initializer is accepted with neither an instance nor storage, and its seat is shared", () => {
  const graph = parse(sharedGraph());
  // The borrower's declared name is the lender's initializer name as is (in IR v2 the name is the
  // key of the instance).
  assertEquals(graph.initializers[SHARED_NAME], { shared: true });
  validateGraphContracts(graph);
  // The seat is "borrowed". Only the borrower-side consumption (eligibility + channel axis 0)
  // determines the lender seat it expects; it is matched against the lender's codec when the
  // borrower Session is built.
  assertEquals(planWeightResidency(graph).get(SHARED_NAME), {
    seat: "shared",
    eligible: true,
    i4Eligible: true,
    i2Eligible: true,
    consumerAxis: 0,
  });
});

test("a shared initializer that writes an instance key or storage is rejected", () => {
  // An IR v2 initializer declaration can contain only `shared` (the mapping to the instance lives
  // in the container's binding table). A form that writes the old distribution form's `tensor` /
  // `storage` as is fails as an unknown key.
  for (const key of ["tensor", "storage"]) {
    const graph = sharedGraph();
    Object.assign(graph.initializers[SHARED_NAME], { [key]: "model.lm_head.weight" });
    assertThrows(() => parse(graph), IrError, `unknown key '${key}'`);
  }
  // `shared` can only be `true` (`false` would be the same declaration as the field being absent —
  // canonical serialization would split).
  const notTrue = sharedGraph();
  Object.assign(notTrue.initializers[SHARED_NAME], { shared: false });
  assertThrows(() => parse(notTrue), IrError, "write true only when it is borrowed");
});
