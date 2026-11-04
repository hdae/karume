// Checks of the GPU timing diagnostics (timestamp-query, per-op breakdown — ADR 0021).
//
// "Timing does not change the results" and "the breakdown matches the execution itself" are pinned
// by **cross-checking** against the other statistics of the run (SubmitStats.dispatchCount /
// pipelineCount). Looking at the breakdown alone would be green with an empty or zero-entry
// aggregate (a silently dead measurement would slip through).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertGreater, assertThrows } from "@std/assert";
import { elementwiseKey } from "../src/codegen/elementwise.ts";
import { acquireGpu, GpuFeatureError, planTimestampFeature } from "../src/gpu/device.ts";
import type { GpuTimingStats } from "../src/gpu/submit.ts";
import { gemmRowsGeometry, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { matmulKey } from "../src/kernels/matmul.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, f32Bytes, fill, openGraphModel } from "./helpers/model-fixture.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

/**
 * y = relu(x·w + b) + b. **add runs twice** so that the aggregation by key (4 dispatches → 3 keys
 * in the breakdown) can be confirmed by the counts alone.
 */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add", "relu"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [6, 4] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: [6, 3] },
    g: { dtype: "f32", shape: [6, 3] },
    r: { dtype: "f32", shape: [6, 3] },
    y: { dtype: "f32", shape: [6, 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["g"], attrs: {} },
    { op: "relu", ins: ["g"], outs: ["r"], attrs: {} },
    { op: "add", ins: ["r", "b"], outs: ["y"], attrs: {} },
  ],
};

const W = [0.5, -1, 0.25, 2, 0.125, -0.5, -3, 1.5, 0.75, 1, -0.25, 0.5];
const B = [1, -2, 0.5];

/** The pipeline keys the graph actually hits (taken from the same builders as the executor). */
const expectedKeys = (profile: GeometryProfile): ReadonlySet<string> =>
  new Set([
    // x[6,4] × w[4,3]: n = 3 is not a multiple of 4, so the scalar variant runs, and m = 6 hits
    // the small-M tile geometry bucket (`gemmGeometryForRows` in src/kernels/gemm-geometry.ts)
    matmulKey(false, 6, gemmRowsGeometry(profile, 6)),
    elementwiseKey({ op: "add", rank: 2, dtype: "f32" }),
    elementwiseKey({ op: "relu", rank: 2, dtype: "f32" }),
  ]);

/**
 * The output's **bytes** (that timing mode does not change the results is checked on bits, not
 * values).
 */
const outputBytes = (tensor: Tensor): Uint8Array =>
  new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength).slice();

/** Runs the graph once and returns the output bytes and the diagnostics. */
const runOnce = async (
  gpuTiming: boolean | undefined,
): Promise<{
  readonly timingEnabled: boolean;
  readonly bytes: Uint8Array;
  readonly features: ReadonlySet<string>;
  readonly dispatchCount: number;
  readonly pipelineCount: number;
  readonly timing: GpuTimingStats | undefined;
  readonly profile: GeometryProfile;
}> => {
  const gpu = await acquireGpu(gpuTiming === undefined ? {} : { gpuTiming });
  const opened = await openGraphModel(GRAPH, [
    { graph: "model", initializer: "w", bytes: f32Bytes(W), encoding: { codec: "f32" } },
    { graph: "model", initializer: "b", bytes: f32Bytes(B), encoding: { codec: "f32" } },
  ]);
  const session = await createSessionFromContainer(gpu, opened, "model");
  try {
    const outputs = await session.run({ x: fill([6, 4], (i) => ((i % 11) - 5) * 0.5) });
    const diagnostics = session.diagnostics();
    return {
      timingEnabled: gpu.gpuTimingEnabled,
      bytes: outputBytes(outputs["y"]),
      features: new Set(gpu.device.features),
      dispatchCount: diagnostics.submit.dispatchCount,
      pipelineCount: diagnostics.pipelineCount,
      timing: diagnostics.lastRunTiming,
      profile: deviceGeometryProfile(gpu),
    };
  } finally {
    await session.dispose();
    gpu.destroy();
  }
};

// Of the three values, only "true × feature absent" cannot be produced on real hardware (an
// adapter's features cannot be removed), so all 6 combinations are pinned on the pure function the
// decision was extracted into.
test("planTimestampFeature decides the three values of gpuTiming and fails loudly when a required feature is missing", () => {
  const present = new Set(["shader-f16", "timestamp-query"]);
  const absent = new Set(["shader-f16"]);

  // MUST: `undefined` means "do not request" (the same rule as shaderF16). Turning it back into
  // "request when present" would put the cost of timing (1 dispatch = 1 pass) on the default path
  // depending on the adapter.
  assertEquals(
    planTimestampFeature(present, undefined),
    false,
    "the default does not request it even when present",
  );
  assertEquals(planTimestampFeature(absent, undefined), false);
  assertEquals(planTimestampFeature(present, true), true);
  assertEquals(
    planTimestampFeature(present, false),
    false,
    "false does not request it even when present",
  );
  assertEquals(planTimestampFeature(absent, false), false);

  const error = assertThrows(
    () => planTimestampFeature(absent, true),
    GpuFeatureError,
    "timestamp-query",
  );
  // The missing feature must be readable (the message also says what the environment offers).
  assert(error.message.includes("shader-f16"), error.message);
});

test.skipIf(!GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE)(
  "the breakdown of gpuTiming: true agrees with the dispatch count and the key set (real GPU)",
  async () => {
    const result = await runOnce(true);
    assert(
      result.timingEnabled,
      "requesting true and not getting it enabled is a contradiction on the device side",
    );
    const timing = result.timing;
    assert(timing !== undefined, "with timing enabled the breakdown is always there");

    assertEquals(result.dispatchCount, 4, "matmul + add + relu + add");
    assertEquals(
      timing.dispatchCount,
      result.dispatchCount,
      "cross-check of the breakdown's total dispatch count",
    );
    assertEquals(
      new Set(timing.entries.map((entry) => entry.key)),
      expectedKeys(result.profile),
      "the key set matches the pipelines that ran",
    );
    assertEquals(timing.entries.length, result.pipelineCount, "every pipeline created ran");

    for (const entry of timing.entries) {
      assert(entry.ns >= 0, `${entry.key}: negative ns (${entry.ns})`);
      assertGreater(entry.workgroupCount, 0, `${entry.key}: workgroup count`);
    }
    assertEquals(
      timing.entries.find((entry) => entry.key.includes(":add:"))?.dispatchCount,
      2,
      "dispatches of the same key are aggregated",
    );
    assertEquals(
      timing.totalNs,
      timing.entries.reduce((total, entry) => total + entry.ns, 0),
      "the total is the sum of the breakdown",
    );
    assertEquals(
      [...timing.entries].sort((a, b) => b.ns - a.ns).map((entry) => entry.key),
      timing.entries.map((entry) => entry.key),
      "the table is in descending order of ns",
    );
    // Negative differences clamped to 0 (a non-monotonic driver) are not swallowed but observed as
    // a count.
    assertEquals(timing.clampedNegativeSamples, 0, "no non-monotonic timestamps are observed");
    assertGreater(timing.totalNs, 0, "all-zero timings on a real GPU mean the measurement is dead");
  },
);

// The check that the default (undefined) and an explicit false are **the same**. If they diverge,
// the cost of timing lands on the default path depending on the adapter (1 dispatch = 1 pass —
// ADR 0021 / option 3).
test.skipIf(!GPU_AVAILABLE)(
  "gpuTiming's default and false request no feature and produce no breakdown (real GPU)",
  async () => {
    for (const requested of [undefined, false] as const) {
      const result = await runOnce(requested);
      const where = `gpuTiming: ${String(requested)}`;
      assertEquals(
        result.features.has("timestamp-query"),
        false,
        `${where}: does not have the feature`,
      );
      assertEquals(result.timingEnabled, false, where);
      assertEquals(result.timing, undefined, where);
    }
  },
);

test.skipIf(!GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE)(
  "timing mode does not change the results (the same graph's outputs match byte for byte, real GPU)",
  async () => {
    const timed = await runOnce(true);
    const plain = await runOnce(false);
    assert(
      timed.timingEnabled,
      "the control does not hold (timing is not enabled on the timed side)",
    );
    assertEquals(
      timed.bytes,
      plain.bytes,
      "opening to 1 dispatch = 1 pass leaves the output bytes identical",
    );
    assertEquals(
      timed.dispatchCount,
      plain.dispatchCount,
      "the dispatch count does not change either",
    );
  },
);
