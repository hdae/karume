import { test } from "vite-plus/test";
import { assert, assertEquals, assertInstanceOf, assertRejects } from "@std/assert";
import { acquireGpu, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import {
  GpuValidationError,
  popFailureScopes,
  pushFailureScopes,
  withPipelineScope,
} from "../src/gpu/error-scope.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { applyReferenceOp } from "../src/reference/ops.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { fill, openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const wgslWithWorkgroupSize = (size: number): string => `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x] + 1.0; }
}
`;

test.skipIf(!GPU_AVAILABLE)(
  "a pipeline that exceeds the workgroup size becomes an exception through errorScope (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache = new PipelineCache(gpu.device);
    try {
      const tooLarge = gpu.limits.maxComputeWorkgroupSizeX + 1;

      // Creation itself raises no synchronous exception (the entry to a silent no-op). Only errorScope makes it visible.
      let threwSynchronously = false;
      gpu.device.pushErrorScope("validation");
      try {
        const module = gpu.device.createShaderModule({ code: wgslWithWorkgroupSize(tooLarge) });
        gpu.device.createComputePipeline({ layout: "auto", compute: { module } });
      } catch {
        threwSynchronously = true;
      }
      const raw = await gpu.device.popErrorScope();
      assert(
        threwSynchronously || raw !== null,
        "an invalid pipeline went through completely undetected",
      );

      // Through PipelineCache it always becomes an exception (the net that catches silent no-ops)
      const error = await assertRejects(
        () => cache.get("test:too-large", wgslWithWorkgroupSize(tooLarge)),
        GpuValidationError,
        "test:too-large",
      );
      assert(error.message.length > "test:too-large".length, "the cause message is kept");
      assertEquals(cache.size, 0, "a failed pipeline is not cached");

      // The previous failure left no errorScope behind (if later checks were absorbed by the wrong scope,
      // validation errors from then on would be invisible for good)
      const valid = await cache.get("test:valid", wgslWithWorkgroupSize(64));
      assert(valid !== undefined);
      assertEquals(cache.size, 1);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "withPipelineScope pops both scopes even on a synchronous exception from body (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // A scope left behind / an excess pop cannot be told from the scope's contents (one's own push and
      // pop always pair up LIFO), so it is checked indirectly with **an outer sentinel scope**. One error
      // is put into the sentinel first, and the test checks that it can be taken back after
      // withPipelineScope.
      const { layout } = await new PipelineCache(gpu.device).get(
        "pipeline-scope-probe",
        wgslWithWorkgroupSize(64),
      );
      gpu.device.pushErrorScope("validation");
      // A bindGroup missing binding 0 raises no synchronous exception and shows up only in errorScope
      gpu.device.createBindGroup({ layout, entries: [] });

      const thrown = new Error("failure on the body side");
      const caught = await assertRejects(() =>
        withPipelineScope(gpu.device, "body-throws", () => {
          throw thrown;
        }),
      );
      assertEquals(caught, thrown, "the body's exception propagates as is");

      // If only 1 scope had been popped, this pop would take the empty scope left behind (= null).
      // If 3 had been popped, the sentinel would already be gone and this pop itself would fail.
      const sentinel = await gpu.device.popErrorScope();
      assert(
        sentinel !== null,
        "the sentinel scope's error cannot be taken back = withPipelineScope left a scope behind",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "an invalid pipeline comes back as validation, not internal (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache = new PipelineCache(gpu.device);
    try {
      // An internal error itself cannot be produced on a real GPU, so this only checks that, with the two
      // scopes introduced, "what was validation does not turn into internal" (detects a wrong priority).
      const tooLarge = gpu.limits.maxComputeWorkgroupSizeX + 1;
      const error = await assertRejects(
        () => cache.get("test:internal-misfire", wgslWithWorkgroupSize(tooLarge)),
        GpuValidationError,
      );
      assertEquals(error.name, "GpuValidationError", "the internal branch misfires");
      assertEquals(cache.size, 0, "a failed pipeline is not cached");

      // Both scopes are popped on the failure path too (a scope left behind would make later checks be
      // absorbed by the wrong scope). Confirmed indirectly by the normal creation right after succeeding.
      const valid = await cache.get("test:internal-misfire-after", wgslWithWorkgroupSize(64));
      assert(valid !== undefined);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "concurrent runs of several Sessions on the same device complete without interfering with each other (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    // Cleanup is guaranteed from right after acquisition. Construction and dispose wait with allSettled
    // for everything to settle before collecting, so a failure on one side does not fail to destroy the
    // Session or device of the side that succeeded (so that a failing test does not carry resources over
    // to later GPU tests).
    try {
      // A run's scope range is serialized by a per-device lock. If a layer called from inside the lock
      // (PipelineCache, etc.) took the lock again it would self-deadlock, so concurrent runs completing
      // without hanging is itself what this regression test covers.
      const built = await Promise.allSettled(
        (["relu", "neg"] as const).map(
          async (op) =>
            await createSessionFromContainer(
              gpu,
              await openGraphModel(singleOpDeclaration(op, [[8, 8]], [[8, 8]])),
              "model",
            ),
        ),
      );
      const sessions = built.flatMap((result) =>
        result.status === "fulfilled" ? [result.value] : [],
      );
      let failure: { readonly error: unknown } | undefined;
      try {
        for (const result of built) if (result.status === "rejected") throw result.reason;
        const x = fill([8, 8], (i) => (i % 13) - 6);

        // Issued without an await in between (the form in which the ranges of the Sessions overlap). The
        // runs also wait with allSettled for everything to settle —— with Promise.all, one reject would move
        // on to dispose while the other run is still running.
        const ran = await Promise.allSettled(sessions.map((session) => session.run({ x0: x })));
        const [relu, neg] = ran.map((result) => {
          if (result.status === "rejected") throw result.reason;
          return result.value;
        });

        assertEquals([...relu.y.data], [...applyReferenceOp("relu", [x]).data]);
        assertEquals([...neg.y.data], [...applyReferenceOp("neg", [x]).data]);
      } catch (error) {
        failure = { error };
      }
      // Throw the body's failure first (a dispose failure does not overwrite the cause).
      const disposed = await Promise.allSettled(sessions.map((session) => session.dispose()));
      if (failure !== undefined) throw failure.error;
      for (const result of disposed) if (result.status === "rejected") throw result.reason;
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "Session.run guards its own scope range with the per-device lock (real GPU)",
  async () => {
    // The contract of the primitive (GpuContext.withScopeLock) is pinned by gpu_device_test, but the
    // wiring that Session.run **uses** it is pinned only here. Without this test, a refactor that drops
    // the wrapper from the executor (or moves readback / flush outside the lock) would pass all green,
    // and the misattribution that m0-review rated major would come back as is.
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openGraphModel(singleOpDeclaration("relu", [[8, 8]], [[8, 8]])),
      "model",
    );
    try {
      const { layout } = await new PipelineCache(gpu.device).get(
        "run-scope-lock-probe",
        wgslWithWorkgroupSize(64),
      );
      const x = fill([8, 8], (i) => (i % 13) - 6);

      // MUST: let the faulty side open **first**. pop takes the top of the stack unconditionally, so the
      // misattribution happens only with the overlap "faulty pushes first and pops first" (with the run
      // first, both are attributed to the right side, and the test would stay green even without the
      // lock).
      const faulty = gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
        pushFailureScopes(gpu.device);
        // A bindGroup missing binding 0 raises no synchronous exception and shows up only in errorScope
        gpu.device.createBindGroup({ layout, entries: [] });
        for (let i = 0; i < 4; i += 1) await Promise.resolve();
        return await popFailureScopes(gpu.device, "faulty");
      });
      const running = session.run({ x0: x });

      const [faultyResult, outputs] = await Promise.all([faulty, running]);

      assertInstanceOf(
        faultyResult,
        GpuValidationError,
        "the side that caused the error captures it",
      );
      assert(faultyResult.message.startsWith("faulty:"), faultyResult.message);
      // Without the lock the run fails with faulty's error (a run that should pass fails with someone
      // else's message), and faulty misses its own failure.
      assertEquals([...outputs.y.data], [...applyReferenceOp("relu", [x]).data]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);
