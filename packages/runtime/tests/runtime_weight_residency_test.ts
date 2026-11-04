// The check of the residency planner (src/runtime/weight-residency.ts). No GPU is used at all (it
// opens containers — the goldens' `krm` / in-memory containers of synthetic graphs — and looks at
// the residency classes and byte counts decided by the declarations alone).
//
// Two things are checked:
// ① **The residency class** (whether each weight lands on f16 / i8 / i4 residency, CPU expansion
//    or raw-byte residency) — the expected values are hand-written constants per fixture.
//    Rebuilding them with the planner's own predicates would make the test a tautology.
// ② **Byte counts from the declarations = byte counts of the container's supply plan** (both
//    payload and scale) — the planner counts from the declared shapes and the residency classes
//    alone, without looking at the binding table or the table of contents, so this is the only
//    comparison showing that "the declarations and the real data point to the same numbers". For
//    the i8 scale, **a swapped channel axis immediately shows up as a different byte count** (the
//    `[Cin,Cout,K]` of conv_transpose1d is axis 1 — read as axis 0 it does not match the golden's
//    real scale).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type InitializerSupply, mergedGraph } from "../src/format/container/bind.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planWeightResidency, type WeightResidency } from "../src/runtime/weight-residency.ts";
import { openSeriesContainer } from "./helpers/container-files.ts";
import { autoTensors, type StorageMap } from "./helpers/merged-graph.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

const GOLDEN_ROOT = new URL("./fixtures/golden/", import.meta.url);
/** The representative path of one golden's container (`MODEL_FILE` of `goldens.py`). */
const MODEL_FILE = "model.krm";

/**
 * Just the surface needed to compare residency classes and byte counts (the merged graph and the
 * supply plan).
 */
type ResidencyTarget = {
  readonly graph: IrGraph;
  readonly supplies: ReadonlyMap<string, InitializerSupply>;
};

/**
 * Opens one golden's **container** and returns the merged graph and the supply plan (graph name =
 * the name of its directory — `graph_name=spec.name` of `goldens.py`).
 *
 * This goes through neither the GPU nor a Session, so it opens with the bare merge surface instead
 * of `prepareContainer`. Only the header and the 2 documents are read; not a single byte of a
 * block.
 */
const openGoldenContainer = async (model: string): Promise<ResidencyTarget> => {
  const opened = await openSeriesContainer(new URL(`${model}/${MODEL_FILE}`, GOLDEN_ROOT));
  const bound = opened.graphs[model];
  assert(
    bound !== undefined,
    `the container has no graph '${model}' (it has ${Object.keys(opened.graphs).join(" / ")})`,
  );
  return { graph: mergedGraph(bound, model), supplies: bound.supplies };
};

/**
 * Opens a synthetic graph as an **in-memory container** (the same merge surface without writing a
 * `krm`). The supplies' contents are not looked at, so zeros of the length decided by the declared
 * shape and the codec suffice (helpers/merged-graph.ts).
 */
const openGraph = (graph: DeclarationJson, storage: StorageMap = {}): ResidencyTarget => {
  const bound = memoryModel(graph, autoTensors(graph, storage)).graphs[GRAPH_NAME];
  return { graph: mergedGraph(bound, GRAPH_NAME), supplies: bound.supplies };
};

/**
 * Name → residency class (the comparison with the expected values looks only at the class — the
 * byte counts are checked separately against the real data).
 */
const seats = (model: { readonly graph: IrGraph }): Record<string, WeightResidency["seat"]> =>
  Object.fromEntries(
    [...planWeightResidency(model.graph)].map(([name, plan]) => [name, plan.seat]),
  );

/**
 * Compares the byte counts from the declarations with **the container's supply plan** (both
 * payload and scale).
 *
 * The residency planner (`weight-residency.ts`) and the container's merge layer
 * (`format/container/bind.ts`) derive the same quantities **independently** — the former from the
 * IR's declared shapes and the residency classes, the latter from the binding table's encoding and
 * the table of contents. So this compares 2 implementations and is not a tautology. A swapped
 * channel axis (the `[Cin,Cout,K]` of conv_transpose1d has row axis 1) shows up as a different
 * scale byte count.
 *
 * NOTE: the channel-axis claim holds **only on the golden side**, though. `autoTensors`, which
 * builds the synthetic graphs' supplies, fixes rowAxis 0 for the scale length without looking at
 * the consuming op, so the synthetic-side comparison only amounts to "counted twice with the same
 * axis from the declared shape".
 */
const assertDeclaredBytesMatchContainer = (golden: ResidencyTarget): void => {
  const { graph, supplies } = golden;
  const plan = planWeightResidency(graph);
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    const seat = plan.get(name);
    assert(seat !== undefined, `initializer '${name}' has no residency class`);
    // A shared declaration (borrowed — ADR 0096 stage 2 §1.3) has no data in the container, so it
    // is outside this helper (the check is that the assets it covers have none).
    assert(initializer.shared === undefined, `initializer '${name}' is a shared declaration`);
    assert(seat.seat !== "shared", `initializer '${name}' has the residency class shared`);
    const supply = supplies.get(name);
    assert(supply !== undefined, `initializer '${name}' has no supply plan`);
    // Data split into pieces arrives as a sequence of blocks (container-v1 §5), so the payloads are
    // summed.
    const payloadBytes = supply.blocks.reduce((total, block) => total + block.payloadBytes, 0);
    assertEquals(seat.payloadBytes, payloadBytes, `payload byte count of ${name}`);
    if (seat.seat !== "i8" && seat.seat !== "i4") continue;
    assert(supply.scale !== undefined, `initializer '${name}' has no scale supply`);
    assertEquals(seat.scaleBytes, supply.scale.payloadBytes, `scale byte count of ${name}`);
  }
};

// ---------------------------------------------------------------------------
// Comparison with the existing fixtures (distribution forms written by the real exporter)
// ---------------------------------------------------------------------------

test("golden `mlp`: every uncompressed storage gets the raw-byte residency class", async () => {
  const model = await openGoldenContainer("mlp");
  assertEquals(seats(model), { w1: "raw", b1: "raw", w2: "raw", b2: "raw" });
  assertDeclaredBytesMatchContainer(model);
});

test("golden `i8_weights`: every i8 consumed through a weight slot is i8-resident (bias stays f32)", async () => {
  const model = await openGoldenContainer("i8_weights");
  assertEquals(seats(model), {
    // the weight slots of embedding / linear / conv1d / conv_transpose1d / conv2d
    "table.weight": "i8",
    "dense.weight": "i8",
    "conv.weight": "i8",
    "up.weight": "i8",
    "image.weight": "i8",
    // bias is a slot outside the eligibility test (so the "bias drags the weight down into
    // demotion" case that ADR 0006 named does not arise), so it stays f32 with raw-byte residency
    "dense.bias": "raw",
    "conv.bias": "raw",
    "up.bias": "raw",
    "image.bias": "raw",
  });
  // conv_transpose1d's `up.weight` is `[Cin,Cout,K]` = [5,2,3], with channel axis **1**.
  // Mistaken for axis 0, the scale becomes 5 elements (20 bytes) and misses the container side's 2
  // elements (8 bytes).
  assertDeclaredBytesMatchContainer(model);
});

test("golden `conv_transpose` / `embedding_lookup`: byte counts from the declarations match the container", async () => {
  assertDeclaredBytesMatchContainer(await openGoldenContainer("conv_transpose"));
  assertDeclaredBytesMatchContainer(await openGoldenContainer("embedding_lookup"));
});

// ---------------------------------------------------------------------------
// Residency-class branches (eligible / ineligible)
// ---------------------------------------------------------------------------

/**
 * A graph that can put **the same storage** on a linear weight (eligible) and a mul operand
 * (ineligible). IR v2 declarations carry no storage, so the codec of `w` / `g` is decided by the
 * supply side (the second argument of `openGraph`).
 */
const twoPathGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 3] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
    g: {},
  },
  values: {
    w: { dtype: "f32", shape: [3, 3] },
    b: { dtype: "f32", shape: [3] },
    g: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: [2, 3] },
    y: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [
    { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "g"], outs: ["y"], attrs: {} },
  ],
});

test("f16: consumption through weight slots only gets the f16 class, other consumption the CPU-expansion class", () => {
  const model = openGraph(twoPathGraph(), { w: "f16", g: "f16" });
  const plan = planWeightResidency(model.graph);
  assertEquals(seats(model), { w: "f16", b: "raw", g: "expanded" });
  // The f16 payload is numel×2 (alignment padding is the upload side's business, so it does not
  // appear in the residency class)
  assertEquals(plan.get("w"), { seat: "f16", payloadBytes: 18 });
  // After expansion it is f32 numel×4 — held in a field separate from the declared payload (6
  // bytes)
  assertEquals(plan.get("g"), { seat: "expanded", payloadBytes: 6, expandedBytes: 12 });
  assertDeclaredBytesMatchContainer(model);
});

test("i8: the residency class carries the scale byte count and the channel axis", () => {
  const model = openGraph(twoPathGraph(), { w: "int8-sym", g: "f16" });
  // A linear weight is `[out,in]`, so the row axis is 0 → the scale is 3 elements = 12 bytes
  assertEquals(planWeightResidency(model.graph).get("w"), {
    seat: "i8",
    payloadBytes: 9,
    scaleBytes: 12,
    rowAxis: 0,
  });
  assertDeclaredBytesMatchContainer(model);
});

/** A graph of a single `conv1d(x, w, b)` (w is i4 + a rank-2 group scale). */
const i4Conv1dGraph = (groups: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 32, 6] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
  },
  values: {
    // row length = Cin/groups · K (with groups == 1, 32·2 = 64 = four g16 groups)
    w: { dtype: "f32", shape: [4, 32 / groups, 2] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 5] },
  },
  nodes: [
    {
      op: "conv1d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: 1, padding: 0, dilation: 1, groups },
    },
  ],
});

const i4Conv1dModel = (groups: number): ResidencyTarget =>
  openGraph(i4Conv1dGraph(groups), { w: { codec: "int4-sym-g", groupSize: 16 } });

test("i4: conv1d(groups==1), which has a dequantizing path, gets the i4 class; groups>1 gets the CPU-expansion class", () => {
  const direct = i4Conv1dModel(1);
  // numel 256 → packed 128 bytes / the group scale is [4, 4], 16 elements = 64 bytes
  assertEquals(planWeightResidency(direct.graph).get("w"), {
    seat: "i4",
    payloadBytes: 128,
    scaleBytes: 64,
    groupSize: 16,
  });
  assertDeclaredBytesMatchContainer(direct);

  // groups > 1 goes to the direct kernel = there is no dequantizing path, so it falls back to CPU
  // expansion
  const grouped = i4Conv1dModel(2);
  // numel 128 → packed 64 bytes, 128×4 = 512 bytes after expansion
  assertEquals(planWeightResidency(grouped.graph).get("w"), {
    seat: "expanded",
    payloadBytes: 64,
    expandedBytes: 512,
  });
  assertDeclaredBytesMatchContainer(grouped);
});

test("an initializer that is a graph output is not resident in compressed form (readback assumes f32)", () => {
  const graph = twoPathGraph();
  const model = openGraph({ ...graph, outputs: ["y", "w"] }, { w: "f16", g: "f16" });
  assertEquals(seats(model)["w"], "expanded");
});

// ---------------------------------------------------------------------------
// fail loudly
// ---------------------------------------------------------------------------

test("an i8 whose consumers disagree on the channel axis fails without getting a residency class", () => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "conv_transpose1d"] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: [2, 3] },
      { name: "z", dtype: "f32", shape: [1, 3, 4] },
    ],
    outputs: ["h", "u"],
    initializers: {
      // linear requires axis 0, conv_transpose1d axis 1 (ADR 0019)
      w: {},
      b: {},
      c: {},
    },
    values: {
      w: { dtype: "f32", shape: [3, 3] },
      b: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      h: { dtype: "f32", shape: [2, 3] },
      u: { dtype: "f32", shape: [1, 3, 5] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
      {
        op: "conv_transpose1d",
        ins: ["z", "w", "c"],
        outs: ["u"],
        attrs: { stride: 1, padding: 0 },
      },
    ],
  };
  // The channel axis is derived from the consuming ops, so it fails before the residency plan
  // decides the class (the merge layer does not look at the consumers, so the failure comes from
  // `planWeightResidency`, not `openGraph`).
  const model = openGraph(graph, { w: "int8-sym" });
  assertThrows(
    () => planWeightResidency(model.graph),
    ExecutionError,
    "consumers disagree on the channel axis",
  );
});
