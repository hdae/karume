// `SessionOptions.fullStateStorage` without a GPU: the spelling, the per-slot decision, and the bytes
// and temporaries the estimate claims. What a real context allocates and what a real run computes is
// in gpu_full_state_storage_test.ts.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { slotStateStorage } from "../src/kernels/state-storage.ts";
import type { EstimateOptions } from "../src/runtime/estimate.ts";
import { stateSlotBytes } from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  assertExecutionKnobs,
  sessionOptionsViolation,
} from "../src/runtime/session-options-check.ts";
import { bothScenarios, openGraph } from "./helpers/estimate-graphs.ts";
import { fullStateGraph } from "./helpers/full-state-graph.ts";

const LENGTHS = 8;
const CAPACITY = 64;
/** `[1, 2, 64, 8]` in f32. Written out so the expected values do not come from the code under test. */
const SLOT_F32 = 4096;

const estimate = (
  graph: ReturnType<typeof fullStateGraph>,
  options: Omit<EstimateOptions, "generation"> = {},
) =>
  openGraph(graph).estimate({
    generation: { chunkLength: 4, bindings: { C: CAPACITY } },
    maxStorageBufferBindingSize: 1 << 30,
    ...options,
  });

const SHAPE = { heads: 4, kvHeads: 2, depth: 8 };

describe("fullStateStorage spelling", () => {
  it("the default and both values are accepted", () => {
    assertEquals(sessionOptionsViolation({}), undefined);
    assertEquals(sessionOptionsViolation({ fullStateStorage: "f32" }), undefined);
    assertEquals(sessionOptionsViolation({ fullStateStorage: "f16" }), undefined);
  });

  it("another spelling is a violation that lists what is accepted", () => {
    const violation = sessionOptionsViolation({ fullStateStorage: "F16" as "f16" });
    assert(violation?.includes("fullStateStorage") === true, `${violation}`);
    assert(violation?.includes("'f32' / 'f16'") === true, `${violation}`);
  });

  it("a value that only converts to an accepted key is a violation (['f16'] is not 'f16')", () => {
    assertNotEquals(
      sessionOptionsViolation({ fullStateStorage: ["f16"] as unknown as "f16" }),
      undefined,
    );
  });

  it("the knob check names it with the other knobs", () => {
    assertThrows(
      () =>
        assertExecutionKnobs("f32", "f32", "f32", "f32", "sequential", "sequential", "f8" as "f16"),
      ExecutionError,
      'fullStateStorage: "f8"',
    );
  });

  it("the estimate refuses the same spelling", () => {
    assertThrows(
      () =>
        estimate(fullStateGraph({ ...SHAPE, slots: "both" }), { fullStateStorage: "F16" as "f16" }),
      ExecutionError,
      "fullStateStorage",
    );
  });
});

describe("the storage of one slot", () => {
  it("only a full slot under f16 is f16", () => {
    assertEquals(slotStateStorage("f16", false), "f16");
    assertEquals(slotStateStorage("f16", true), "f32");
    assertEquals(slotStateStorage("f32", false), "f32");
    assertEquals(slotStateStorage("f32", true), "f32");
  });

  it("the bytes follow that decision", () => {
    const shape = [1, 2, 64, 8];
    assertEquals(stateSlotBytes("s", shape, false, "f32"), SLOT_F32);
    assertEquals(stateSlotBytes("s", shape, false, "f16"), SLOT_F32 / 2);
    assertEquals(stateSlotBytes("s", shape, true, "f16"), SLOT_F32);
  });

  it("an odd D fails only where it would be packed", () => {
    const odd = [1, 2, 64, 7];
    assertThrows(() => stateSlotBytes("kf", odd, false, "f16"), CodegenError, "state 'kf'");
    assertEquals(stateSlotBytes("kf", odd, false, "f32"), 1 * 2 * 64 * 7 * 4);
    assertEquals(stateSlotBytes("ks", odd, true, "f16"), 1 * 2 * 64 * 7 * 4);
  });
});

describe("the estimate of the state bytes", () => {
  it("f16 halves exactly the full slots and leaves the sliding slots alone", () => {
    const graph = fullStateGraph({ ...SHAPE, slots: "both" });
    assertEquals(estimate(graph).resident.stateBytes, 4 * SLOT_F32 + LENGTHS);
    assertEquals(
      estimate(graph, { fullStateStorage: "f32" }).resident.stateBytes,
      4 * SLOT_F32 + LENGTHS,
    );
    assertEquals(
      estimate(graph, { fullStateStorage: "f16" }).resident.stateBytes,
      2 * (SLOT_F32 / 2) + 2 * SLOT_F32 + LENGTHS,
    );
  });

  it("a graph with only sliding slots does not move", () => {
    const graph = fullStateGraph({ ...SHAPE, slots: "sliding" });
    assertEquals(
      estimate(graph, { fullStateStorage: "f16" }).resident.stateBytes,
      2 * SLOT_F32 + LENGTHS,
    );
  });

  it("nothing else in the report moves: S and its statistics stay f32", () => {
    const graph = fullStateGraph({ ...SHAPE, slots: "both" });
    const f32 = bothScenarios(estimate(graph));
    const f16 = bothScenarios(estimate(graph, { fullStateStorage: "f16" }));
    assertEquals(f16.prefill.workspaceBytes, f32.prefill.workspaceBytes);
    assertEquals(f16.decode.workspaceBytes, f32.decode.workspaceBytes);
  });

  it("f16 lets through a slot that only fits the binding limit when halved", () => {
    // A depth large enough for the slot, `[1, 2, 64, 64]` = 32,768 bytes, to outgrow S (4,096 bytes).
    const graph = fullStateGraph({ ...SHAPE, depth: 64, slots: "full" });
    const limits = { maxBufferSize: 16384 };
    assertThrows(() => estimate(graph, limits), ExecutionError, "maxBufferSize");
    assertEquals(
      estimate(graph, { ...limits, fullStateStorage: "f16" }).resident.stateBytes,
      2 * 16384 + LENGTHS,
    );
  });

  it("an odd D returns no number under f16, as the real allocation returns no context", () => {
    const graph = fullStateGraph({ heads: 4, kvHeads: 2, depth: 7, slots: "full" });
    assertThrows(() => estimate(graph, { fullStateStorage: "f16" }), CodegenError, "even depth");
    estimate(graph);
  });
});

describe("the fused stats + PV under f16", () => {
  const decodeBytes = (
    graph: ReturnType<typeof fullStateGraph>,
    options: Omit<EstimateOptions, "generation">,
  ): number => bothScenarios(estimate(graph, options)).decode.workspaceBytes;

  it("an f16 full slot counts the statistics temporary the fused kernel has none of", () => {
    const graph = fullStateGraph({ ...SHAPE, slots: "full" });
    assert(
      decodeBytes(graph, { stateAttentionReduce: "parallel-fused" }) <
        decodeBytes(graph, { stateAttentionReduce: "parallel" }),
      "the fixture does not fuse on f32, so this case would prove nothing",
    );
    assertEquals(
      decodeBytes(graph, { stateAttentionReduce: "parallel-fused", fullStateStorage: "f16" }),
      decodeBytes(graph, { stateAttentionReduce: "parallel", fullStateStorage: "f16" }),
    );
  });

  it("a sliding slot keeps the fusion under f16", () => {
    const graph = fullStateGraph({ ...SHAPE, slots: "sliding" });
    assertEquals(
      decodeBytes(graph, { stateAttentionReduce: "parallel-fused", fullStateStorage: "f16" }),
      decodeBytes(graph, { stateAttentionReduce: "parallel-fused" }),
    );
  });
});
