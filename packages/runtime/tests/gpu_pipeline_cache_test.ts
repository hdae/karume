import { platformObject } from "@karume/test-support/webgpu";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import { GpuInternalError, GpuValidationError } from "../src/gpu/error-scope.ts";
import {
  parseStorageRoles,
  PipelineCache,
  PipelineKeyConflictError,
  SessionPipelines,
  StorageRoleError,
} from "../src/gpu/pipeline-cache.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { fill, openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** A fake holding only the surface PipelineCache touches. The full DOM types are not reproduced, so it is passed with a cast. */
type FakeGpu = {
  readonly device: GPUDevice;
  /** Record of the call order (to observe how the errorScopes are set up). */
  readonly calls: string[];
  readonly modules: string[];
  /** The validation error the next popErrorScope returns. Swapped to produce "failure → success on retry". */
  scopeError: GPUError | null;
  /**
   * The error the next popErrorScope returns from the **internal scope**. The surface for hitting failures that
   * cannot be produced on a real GPU (a shader too complex to compile and the like).
   */
  internalError: GPUError | null;
};

const createFakeGpu = (
  scopeError: GPUError | null = null,
  internalError: GPUError | null = null,
): FakeGpu => {
  const calls: string[] = [];
  const modules: string[] = [];
  // A real device returns only the errors of the scope whose filter matches, so the fake also tracks the pushed
  // filters LIFO and answers per filter (returning the same error to both would misfire the internal-first rule
  // of withPipelineScope).
  const filters: string[] = [];
  const device = {
    pushErrorScope: (filter: string): void => {
      calls.push(`push:${filter}`);
      filters.push(filter);
    },
    popErrorScope: (): Promise<GPUError | null> => {
      calls.push("pop");
      const filter = filters.pop();
      if (filter === "validation") return Promise.resolve(fake.scopeError);
      return Promise.resolve(filter === "internal" ? fake.internalError : null);
    },
    createShaderModule: (descriptor: { readonly code: string }) => {
      calls.push("createShaderModule");
      modules.push(descriptor.code);
      return {
        getCompilationInfo: () =>
          Promise.resolve({
            messages: [{ lineNum: 3, linePos: 7, message: "shader detail" }],
          }),
      };
    },
    createComputePipelineAsync: () => {
      calls.push("createComputePipelineAsync");
      return Promise.resolve({
        id: modules.length,
        getBindGroupLayout: () => {
          calls.push("getBindGroupLayout");
          return { id: modules.length };
        },
      });
    },
  };
  const fake: FakeGpu = {
    device: device as unknown as GPUDevice,
    calls,
    modules,
    scopeError,
    internalError,
  };
  return fake;
};

const WGSL_A = "@compute @workgroup_size(1) fn main() {}";
const WGSL_B = "@compute @workgroup_size(2) fn main() {}";

test("PipelineCache does not regenerate a pipeline for the same key and the same WGSL", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  const first = await cache.get("elementwise:add:f32", WGSL_A);
  const second = await cache.get("elementwise:add:f32", WGSL_A);

  assertStrictEquals(first, second);
  assertEquals(cache.size, 1);
  assertEquals(gpu.modules, [WGSL_A], "the second call does not create a shader module");
});

test("PipelineCache shares a single generation even for requests of the same key that are still pending", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  // Starting 2 without an await in between = the second arrives while the first is still unresolved
  const first = cache.get("elementwise:add:f32", WGSL_A);
  const second = cache.get("elementwise:add:f32", WGSL_A);
  const [firstPipeline, secondPipeline] = await Promise.all([first, second]);

  assertStrictEquals(firstPipeline, secondPipeline);
  assertEquals(cache.size, 1);
  assertEquals(gpu.modules, [WGSL_A], "concurrent requests create the shader module only once");
  assertEquals(
    gpu.calls.filter((call) => call === "createComputePipelineAsync").length,
    1,
    "concurrent requests generate the pipeline only once",
  );
  assertEquals(
    gpu.calls.filter((call) => call === "push:validation").length,
    2,
    "module and layout are scoped separately, without duplication for the same key",
  );
});

test("PipelineCache does not carry over a failed generation and rebuilds on retry", async () => {
  const error = { message: "shader compile failed" } as unknown as GPUError;
  const gpu = createFakeGpu(error);
  const cache = new PipelineCache(gpu.device);

  await assertRejects(
    () => cache.get("retry", WGSL_A),
    GpuValidationError,
    "shader compile failed",
  );
  assertEquals(cache.size, 0, "a failed entry is not kept");

  // If the failure stayed in the cache as a rejected Promise, every later get() would keep returning the same
  // rejection without touching the device (= no recovery from a transient failure)
  gpu.scopeError = null;
  const pipeline = await cache.get("retry", WGSL_A);

  assertEquals(cache.size, 1);
  assertEquals(gpu.modules, [WGSL_A, WGSL_A], "a retry rebuilds");
  assertStrictEquals(
    await cache.get("retry", WGSL_A),
    pipeline,
    "after success it is shared again",
  );
});

test("PipelineCache throws immediately when a different WGSL arrives for the same key (determinism broken)", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  await cache.get("elementwise:add:f32", WGSL_A);
  await assertRejects(
    () => cache.get("elementwise:add:f32", WGSL_B),
    PipelineKeyConflictError,
    "elementwise:add:f32",
  );
  assertEquals(gpu.modules, [WGSL_A], "a conflict does not touch the device");
});

test("PipelineCache wraps pipeline generation in 2 scopes, internal + validation", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);

  await cache.get("k", WGSL_A);

  assertEquals(gpu.calls, [
    "push:internal",
    "push:validation",
    "createShaderModule",
    "createComputePipelineAsync",
    "pop",
    "pop",
    "push:internal",
    "push:validation",
    "getBindGroupLayout",
    "pop",
    "pop",
  ]);
});

test("PipelineCache turns a validation error caught by the errorScope into an exception and does not cache it", async () => {
  const error = { message: "workgroup size exceeds limit" } as unknown as GPUError;
  const gpu = createFakeGpu(error);
  const cache = new PipelineCache(gpu.device);

  await assertRejects(
    () => cache.get("too-large", WGSL_A),
    GpuValidationError,
    "workgroup size exceeds limit",
  );
  assertEquals(cache.size, 0);
});

// GpuInternalError cannot be produced on a real GPU (internal errors are implementation-specific failures), so
// the path that creates this type can only be hit with a fake. The meaning of the public surface is that the
// limits of the environment (internal) and malformed WGSL (validation) branch differently, so the type mapping is pinned here.
test("PipelineCache turns a catch in the internal scope into GpuInternalError", async () => {
  const internal = { message: "shader too complex" } as unknown as GPUError;
  const gpu = createFakeGpu(null, internal);
  const cache = new PipelineCache(gpu.device);

  const error = await assertRejects(() => cache.get("k", WGSL_A), GpuInternalError);
  assert(error.message.includes("k"), error.message);
  assert(error.message.includes("shader too complex"), error.message);
  assertEquals(cache.size, 0, "a failed entry is not kept");

  // The other side of nothing being left over (if both scopes were popped, a normal generation right after passes).
  gpu.internalError = null;
  assert((await cache.get("k", WGSL_A)) !== undefined);
});

// Priority check: later operations that touch a pipeline invalidated by internal raise a **derived** validation
// error, so returning validation first throws away the root-cause internal. Only this form with both at once
// catches a loss of "internal first" (the internal-only case stays green even with the order reverted).
test("PipelineCache returns internal when it catches internal and validation at the same time", async () => {
  const gpu = createFakeGpu(
    { message: "derived: pipeline is invalid" } as unknown as GPUError,
    { message: "shader too complex" } as unknown as GPUError,
  );
  const cache = new PipelineCache(gpu.device);

  const error = await assertRejects(() => cache.get("both", WGSL_A), GpuInternalError);
  assert(error.message.includes("shader too complex"), error.message);
  assert(
    !error.message.includes("derived"),
    `the derived validation overwrites the root cause: ${error.message}`,
  );
  assertEquals(cache.size, 0);
});

test("SessionPipelines shares the one device-lifetime cache and a second Session does not recompile", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);
  const first = new SessionPipelines(cache);
  const second = new SessionPipelines(cache);

  await first.get("elementwise:add:f32", WGSL_A);
  await second.get("elementwise:add:f32", WGSL_A);

  assertEquals(gpu.modules, [WGSL_A], "the second Session does not create a shader module");
  assertEquals(first.usedCount, 1);
  assertEquals(second.usedCount, 1, "even when shared, 'the count I used' covers only its own");
  assertEquals(second.deviceCount, 1, "the device total is the union");
});

test("SessionPipelines' used count is the size of the key set (not the number of lookups)", async () => {
  const gpu = createFakeGpu();
  const use = new SessionPipelines(new PipelineCache(gpu.device));

  await use.get("k1", WGSL_A);
  await use.get("k1", WGSL_A);
  await use.get("k2", WGSL_B);

  assertEquals(use.usedCount, 2);
});

test("SessionPipelines does not count a failed key in the used set", async () => {
  const error = { message: "shader compile failed" } as unknown as GPUError;
  const gpu = createFakeGpu(error);
  const use = new SessionPipelines(new PipelineCache(gpu.device));

  await assertRejects(() => use.get("broken", WGSL_A), GpuValidationError);

  assertEquals(
    use.usedCount,
    0,
    "a key that does not remain on the device is not counted as used by the Session alone",
  );
  assertEquals(use.deviceCount, 0);
});

test("PipelineKeyConflictError is detected across Sessions (same device, same key)", async () => {
  const gpu = createFakeGpu();
  // Determinism breaks when "the same key gets a different WGSL". When the cache was per Session, conflicts
  // between different Sessions were never compared at all, and a later Session could silently run a different
  // kernel (the browser's implicit cache is keyed by the WGSL string, so no warning either).
  const cache = new PipelineCache(gpu.device);
  await new SessionPipelines(cache).get("matmul:v4", WGSL_A);

  await assertRejects(
    () => new SessionPipelines(cache).get("matmul:v4", WGSL_B),
    PipelineKeyConflictError,
    "matmul:v4",
  );
  assertEquals(gpu.modules, [WGSL_A], "a conflict does not touch the device");
});

test.skipIf(!GPU_AVAILABLE)(
  "the pipeline cache lives as long as the device, and keys of a disposed Session remain (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const model = () => openGraphModel(singleOpDeclaration("relu", [[8, 8]], [[8, 8]]));
    const x = fill([8, 8], (i) => (i % 13) - 6);

    try {
      const first = await createSessionFromContainer(gpu, await model(), "model");
      assertEquals(
        first.diagnostics().pipelineCount,
        0,
        "the construction phase does not look up a single pipeline (generation is the miss path of the first run)",
      );
      await first.run({ x0: x });
      const used = first.diagnostics().pipelineCount;
      assert(used > 0, "run creates pipelines");
      assertEquals(
        first.diagnostics().devicePipelineCount,
        used,
        "this device holds only the first Session's",
      );
      await first.dispose();

      const second = await createSessionFromContainer(gpu, await model(), "model");
      assertEquals(
        second.diagnostics().devicePipelineCount,
        used,
        "keys of a disposed Session remain in the device total (the lifetime matches GpuContext)",
      );
      assertEquals(
        second.diagnostics().pipelineCount,
        0,
        "the used count starts from 0 per Session",
      );

      await second.run({ x0: x });

      assertEquals(
        second.diagnostics().pipelineCount,
        used,
        "the used count is the same as the first Session's",
      );
      assertEquals(
        second.diagnostics().devicePipelineCount,
        used,
        "the second Session does not add a single pipeline to the device total = it did not recompile the same WGSL",
      );
      await second.dispose();
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Roles of the storage bindings (WGSL declarations → planning input — ADR 0093 decision 1)
// ---------------------------------------------------------------------------

test("parseStorageRoles takes the read / read-write bindings from the storage declarations of group 0", () => {
  const roles = parseStorageRoles(`
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0)@binding(2)var<storage,read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<storage , read > scale: array<f32>;
  `);
  assertEquals(roles, { reads: new Set([1, 3]), writes: new Set([2]) });
  // A WGSL without a single storage declaration gives an empty set (a uniform-only kernel is legal too).
  assertEquals(parseStorageRoles("@group(0) @binding(0) var<uniform> p: P;"), {
    reads: new Set<number>(),
    writes: new Set<number>(),
  });
});

test("parseStorageRoles fails loudly on a double read / read_write declaration of the same binding", () => {
  assertThrows(
    () =>
      parseStorageRoles(`
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
      `),
    StorageRoleError,
    "binding 1 is declared as both read and read_write",
  );
});

test("PipelineCache.get takes the roles from the WGSL and returns them (fails on conflicting declarations before generation)", async () => {
  const gpu = createFakeGpu();
  const cache = new PipelineCache(gpu.device);
  const { roles } = await cache.get(
    "k",
    "@group(0) @binding(1) var<storage, read> a: array<f32>;\n@group(0) @binding(2) var<storage, read_write> o: array<f32>;",
  );
  assertEquals(roles, { reads: new Set([1]), writes: new Set([2]) });
  await assertRejects(
    () =>
      cache.get(
        "bad",
        "@group(0) @binding(1) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read_write> o: array<f32>;",
      ),
    StorageRoleError,
  );
  assertEquals(gpu.modules.length, 1, "a conflicting WGSL does not create a shader module");
});

test("distinguishes internal / validation rejections of an async compile by type and keeps the cause", async () => {
  for (const reason of ["internal", "validation"] as const) {
    const gpu = createFakeGpu();
    const cause = Object.assign(new Error(`compile ${reason}`), { reason });
    gpu.device.createComputePipelineAsync = () => Promise.reject(cause);
    const cache = new PipelineCache(gpu.device);
    const error = await assertRejects<GpuInternalError | GpuValidationError>(
      () => cache.get("async-error", WGSL_A),
      reason === "internal" ? GpuInternalError : GpuValidationError,
    );
    assertStrictEquals(error.cause, cause);
    assert(error.message.includes("3:7 shader detail"));
    assertEquals(cache.size, 0);
  }
});

test("does not repaint an unknown exception of an async compile as validation", async () => {
  const gpu = createFakeGpu();
  const cause = new RangeError("unexpected rejection");
  gpu.device.createComputePipelineAsync = () => Promise.reject(cause);
  const cache = new PipelineCache(gpu.device);
  const error = await assertRejects(() => cache.get("unexpected", WGSL_A), RangeError);
  assertStrictEquals(error, cause);
  assertEquals(cache.size, 0);
});

test("even when module validation fails, waits for the issued compile to settle before rejecting", async () => {
  const gpu = createFakeGpu({ message: "module validation" } as GPUError);
  const compiling = Promise.withResolvers<GPUComputePipeline>();
  gpu.device.createComputePipelineAsync = () => compiling.promise;
  const cache = new PipelineCache(gpu.device);
  let settled = false;
  const checked = assertRejects(
    () => cache.get("drain", WGSL_A),
    GpuValidationError,
    "module validation",
  ).then(() => {
    settled = true;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assertEquals(settled, false);
  assertEquals(cache.size, 1);
  compiling.reject(
    Object.assign(new Error("derived pipeline failure"), {
      reason: "validation",
    }),
  );
  await checked;
  assertEquals(cache.size, 0);
});

test("does not mix up keys and layouts when async compiles complete in reverse order", async () => {
  const gpu = createFakeGpu();
  const first = Promise.withResolvers<GPUComputePipeline>();
  const second = Promise.withResolvers<GPUComputePipeline>();
  gpu.device.createComputePipelineAsync = (descriptor) =>
    descriptor.label === "first" ? first.promise : second.promise;
  const a = platformObject<GPUComputePipeline>({
    label: "first",
    getBindGroupLayout: () => platformObject<GPUBindGroupLayout>({ label: "first" }),
  });
  const b = platformObject<GPUComputePipeline>({
    label: "second",
    getBindGroupLayout: () => platformObject<GPUBindGroupLayout>({ label: "second" }),
  });
  const cache = new PipelineCache(gpu.device);
  const pa = cache.get("first", WGSL_A),
    pb = cache.get("second", WGSL_B);
  second.resolve(b);
  assertStrictEquals((await pb).pipeline, b);
  first.resolve(a);
  assertStrictEquals((await pa).pipeline, a);
  assertEquals(cache.size, 2);
});

test("keeps the original rejection reason and cause even when fetching the compile diagnostics fails", async () => {
  const gpu = createFakeGpu();
  const cause = Object.assign(new Error("compile internal"), { reason: "internal" });
  gpu.device.createComputePipelineAsync = () => Promise.reject(cause);
  gpu.device.createShaderModule = () =>
    platformObject<GPUShaderModule>({
      label: "diagnostics failure",
      getCompilationInfo: () => Promise.reject(new Error("info unavailable")),
    });
  const cache = new PipelineCache(gpu.device);
  const error = await assertRejects(() => cache.get("diagnostics", WGSL_A), GpuInternalError);
  assertStrictEquals(error.cause, cause);
  assert(error.message.includes("info unavailable"));
  assertEquals(cache.size, 0);
});
