// The test-only switch that disables fusion rules (`DISABLED_FUSION_RULES` —
// src/runtime/session-types.ts) is checked at Session construction against the Session's rule list,
// with the check the fusion pass applies, so a bad value fails before the first run. The check runs
// before any GPU work, so a fake device is enough; the plain side it gives on a real GPU is covered
// by tests/gpu_fusion_twin_test.ts.

import { describe, it } from "vite-plus/test";
import { assertRejects } from "@std/assert";
import { createSessionFromContainer, DISABLED_FUSION_RULES } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import { fakeDevice, fakeGpuContext } from "./helpers/fake-gpu.ts";
import { type DeclarationJson, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";

/** `y = x * sigmoid(x)` (the silu rule's form). */
const SILU_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["sigmoid", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [8] }],
  outputs: ["y"],
  initializers: {},
  values: { s: { dtype: "f32", shape: [8] }, y: { dtype: "f32", shape: [8] } },
  nodes: [
    { op: "sigmoid", ins: ["x"], outs: ["s"], attrs: {} },
    { op: "mul", ins: ["x", "s"], outs: ["y"], attrs: {} },
  ],
};

const construct = async (options: SessionOptions) =>
  await createSessionFromContainer(
    fakeGpuContext(fakeDevice()),
    await openModelBytes(SILU_GRAPH, []),
    GRAPH_NAME,
    options,
  );

describe("Session construction with DISABLED_FUSION_RULES", () => {
  it("rejects a name that is not a name of the rule list, naming it", async () => {
    await assertRejects(
      () => construct({ [DISABLED_FUSION_RULES]: ["silu", "Silu"] }),
      ExecutionError,
      "'Silu'",
    );
  });

  it("rejects a value that is neither 'all' nor an array of names", async () => {
    // A JS caller's value outside the type: the check names it instead of reading it as names.
    const outsideTheType = (value: unknown) =>
      value as SessionOptions[typeof DISABLED_FUSION_RULES];
    for (const value of ["every", ["silu", 3]]) {
      await assertRejects(
        () => construct({ [DISABLED_FUSION_RULES]: outsideTheType(value) }),
        ExecutionError,
        "DISABLED_FUSION_RULES",
      );
    }
  });
});
