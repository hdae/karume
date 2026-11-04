import { captureUncaughtExceptions } from "@karume/test-support/process-events";
import { platformObject } from "@karume/test-support/webgpu";
import { describe, it, test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertGreaterOrEqual,
  assertInstanceOf,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import {
  acquireGpu,
  assertLimitsGranted,
  assertShaderF16Executes,
  assertSubgroup32Executes,
  probeSubgroupLaneMapping,
  GpuDeviceLostError,
  GpuFeatureError,
  GpuLimitError,
  planRequiredLimits,
  planShaderF16Feature,
  planSubgroup32Features,
  readAdapterInfo,
  readAdapterLimits,
  REQUIRED_LIMIT_KEYS,
  type RequiredLimits,
  RUNTIME_INTERNAL,
  SHADER_F16_FEATURE,
} from "../src/gpu/device.ts";
import {
  GpuOutOfMemoryError,
  GpuValidationError,
  popFailureScopes,
  pushFailureScopes,
} from "../src/gpu/error-scope.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { fakeDevice, fakeGpuContext, losableGpuContext } from "./helpers/fake-gpu.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

/** The smallest duck type passed to planRequiredLimits (only the needed surface of GPUSupportedLimits). */
const fakeAdapterLimits = (overrides: Partial<RequiredLimits> = {}): GPUSupportedLimits => {
  const base: RequiredLimits = {
    maxBufferSize: 1024,
    maxStorageBufferBindingSize: 1024,
    maxUniformBufferBindingSize: 256,
    maxStorageBuffersPerShaderStage: 8,
    maxUniformBuffersPerShaderStage: 8,
    maxComputeWorkgroupStorageSize: 32768,
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupSizeX: 1024,
    maxComputeWorkgroupSizeY: 1024,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535,
  };
  return { ...base, ...overrides } as unknown as GPUSupportedLimits;
};

test("planRequiredLimits keeps maxStorageBufferBindingSize at or below maxBufferSize", () => {
  const planned = planRequiredLimits(
    fakeAdapterLimits({ maxBufferSize: 512, maxStorageBufferBindingSize: 4096 }),
  );

  assertEquals(planned.maxStorageBufferBindingSize, 512);
  assertEquals(planned.maxBufferSize, 512);
});

// LIMIT_CAPS (a test-only restriction) is used to "reproduce a machine with core defaults". If caps
// that restrict only maxBufferSize left the binding limit unrestricted, the intended machine would not
// be reproduced and the test would silently miss its purpose.
test("planRequiredLimits does not let the binding limit exceed maxBufferSize when caps restrict only maxBufferSize", () => {
  const adapter = fakeAdapterLimits({
    maxBufferSize: 256 * 1024 * 1024,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
  });

  const planned = planRequiredLimits(adapter, { maxBufferSize: 64 * 1024 * 1024 });

  assertEquals(planned.maxBufferSize, 64 * 1024 * 1024);
  assertEquals(
    planned.maxStorageBufferBindingSize,
    64 * 1024 * 1024,
    "not clamped by the capped maxBufferSize (only the binding limit stays unrestricted)",
  );

  // The existing use that restricts only the binding limit does not move by a single byte (independent of the order caps are applied in).
  const narrowBinding = planRequiredLimits(adapter, {
    maxStorageBufferBindingSize: 32 * 1024 * 1024,
  });
  assertEquals(narrowBinding.maxBufferSize, 256 * 1024 * 1024);
  assertEquals(narrowBinding.maxStorageBufferBindingSize, 32 * 1024 * 1024);
});

test("planRequiredLimits raises the compute limits to the adapter's values", () => {
  const planned = planRequiredLimits(fakeAdapterLimits());

  assertEquals(
    planned.maxComputeInvocationsPerWorkgroup,
    1024,
    "does not fall to the spec default 256",
  );
  assertEquals(
    planned.maxComputeWorkgroupSizeX,
    1024,
    "requested as a limit separate from invocations",
  );
  assertEquals(planned.maxComputeWorkgroupStorageSize, 32768);
  assertEquals(planned.maxComputeWorkgroupsPerDimension, 65535);
});

test("assertLimitsGranted fails loudly on a limit below the request", () => {
  const planned = planRequiredLimits(fakeAdapterLimits());
  const granted = fakeAdapterLimits({ maxComputeInvocationsPerWorkgroup: 256 });

  assertThrows(
    () => assertLimitsGranted(planned, granted),
    GpuLimitError,
    "maxComputeInvocationsPerWorkgroup",
  );
  assertLimitsGranted(planned, fakeAdapterLimits());
});

test.skipIf(!GPU_AVAILABLE)(
  "acquireGpu returns a device with the raised limits (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const key of REQUIRED_LIMIT_KEYS) {
        assertGreaterOrEqual(
          gpu.device.limits[key],
          gpu.limits[key],
          `${key} is below the request`,
        );
      }
      // If it stays at the spec default, raising the compute limits had no effect
      assertGreaterOrEqual(gpu.limits.maxComputeInvocationsPerWorkgroup, 256);
      assertGreaterOrEqual(gpu.limits.maxComputeWorkgroupSizeX, 256);
      assert(gpu.limits.maxStorageBufferBindingSize <= gpu.limits.maxBufferSize);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "readAdapterLimits returns the same values as the limits acquireGpu plans (real GPU)",
  async () => {
    // The side that checks a distribution's requiredLimits before downloading the weights (admission in
    // models) uses this return value and `GpuContext.limits` after acquisition as the same yardstick. If
    // the formulas drifted apart, "passed before the download but fails at build" (or the reverse) would
    // appear.
    const limits = await readAdapterLimits();
    const gpu = await acquireGpu();
    try {
      assertEquals(limits, gpu.limits);
    } finally {
      gpu.destroy();
    }
  },
);

// `AcquireGpuOptions.gpu` — the WebGPU implementation can be passed explicitly (Node applications
// hand over Dawn's `GPU` instead of defining `navigator.gpu`). No real GPU is needed: the fake
// adapter counts the requests, and `navigator.gpu` is made to throw so that any fallback to the
// global is caught.
const fakeGpu = (): { readonly gpu: GPU; readonly requests: () => number } => {
  let requests = 0;
  const limits = Object.fromEntries(REQUIRED_LIMIT_KEYS.map((key) => [key, 1 << 20]));
  const adapter = {
    limits,
    features: new Set<string>(),
    requestDevice: (): Promise<GPUDevice> =>
      Promise.resolve(Object.assign(fakeDevice(), { limits })),
  };
  const gpu = {
    requestAdapter: (): Promise<GPUAdapter> => {
      requests += 1;
      return Promise.resolve(adapter as unknown as GPUAdapter);
    },
  } as unknown as GPU;
  return { gpu, requests: () => requests };
};

/** Runs `body` while any read of `navigator.gpu` throws (the global must not be consulted). */
const withoutNavigatorGpu = async (body: () => Promise<void>): Promise<void> => {
  const original = Object.getOwnPropertyDescriptor(navigator, "gpu");
  Object.defineProperty(navigator, "gpu", {
    get: () => {
      throw new Error("navigator.gpu was read although a GPU instance was injected");
    },
    configurable: true,
  });
  try {
    await body();
  } finally {
    if (original === undefined) Reflect.deleteProperty(navigator, "gpu");
    else Object.defineProperty(navigator, "gpu", original);
  }
};

describe("AcquireGpuOptions.gpu (injecting a WebGPU implementation)", () => {
  it("acquireGpu takes the adapter from the given GPU and does not read navigator.gpu", async () => {
    const fake = fakeGpu();
    await withoutNavigatorGpu(async () => {
      const gpu = await acquireGpu({ gpu: fake.gpu });
      gpu.destroy();
    });
    assertEquals(fake.requests(), 1);
  });

  it("readAdapterLimits uses the same entry point", async () => {
    const fake = fakeGpu();
    await withoutNavigatorGpu(async () => {
      const limits = await readAdapterLimits({ gpu: fake.gpu });
      assertEquals(typeof limits, "object");
    });
    assertEquals(fake.requests(), 1);
  });

  it("without it, falls back to navigator.gpu (GpuUnavailableError if absent)", async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { value: undefined, configurable: true });
    try {
      await assertRejects(() => acquireGpu(), Error, "navigator.gpu does not exist");
    } finally {
      if (original === undefined) Reflect.deleteProperty(navigator, "gpu");
      else Object.defineProperty(navigator, "gpu", original);
    }
  });
});

test.skipIf(!GPU_AVAILABLE)(
  "acquireGpu normalizes features / wgslLanguageFeatures into sets (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      assert(gpu.features instanceof Set);
      assert(
        gpu.wgslLanguageFeatures instanceof Set,
        "falls back to an empty set on implementations without it",
      );
      for (const feature of gpu.features) {
        assertEquals(typeof feature, "string");
      }
    } finally {
      gpu.destroy();
    }
  },
);

// Only "true x feature absent" cannot be produced on real hardware (an adapter's features cannot be
// removed), so all 6 combinations are pinned on the pure function the decision was factored into.
// ADR 0028.
test("planShaderF16Feature treats undefined as 'do not request' and fails loudly when a required feature is missing", () => {
  const present = new Set(["shader-f16", "timestamp-query"]);
  const absent = new Set(["timestamp-query"]);

  // MUST: the **opposite** of gpuTiming's automatic decision. f16 computation changes the numbers, so
  // if the adapter's capabilities decided on or off, "changing the machine would silently change the
  // output".
  assertEquals(
    planShaderF16Feature(present, undefined),
    false,
    "not requested by default (no auto)",
  );
  assertEquals(planShaderF16Feature(absent, undefined), false);
  assertEquals(planShaderF16Feature(present, true), true);
  assertEquals(planShaderF16Feature(present, false), false);
  assertEquals(planShaderF16Feature(absent, false), false);

  const error = assertThrows(
    () => planShaderF16Feature(absent, true),
    GpuFeatureError,
    SHADER_F16_FEATURE,
  );
  // The message must say what is missing and how to work around it (what the environment offers + how to return to the default path).
  assert(error.message.includes("timestamp-query"), error.message);
  assert(error.message.includes("f32"), error.message);
});

test.skipIf(!GPU_AVAILABLE)(
  "acquireGpu({shaderF16}) requests the feature and enables it only when the runtime canary passes (real GPU)",
  async () => {
    // (1) Not requested by default (does not widen the device's capability surface of the existing path by a single bit)
    const plain = await acquireGpu();
    try {
      assertEquals(plain.shaderF16Enabled, false);
      assertEquals(plain.features.has(SHADER_F16_FEATURE), false);
    } finally {
      plain.destroy();
    }
    // (2) false means "do not request it even if available" (a statement of intent)
    const declined = await acquireGpu({ shaderF16: false });
    try {
      assertEquals(declined.shaderF16Enabled, false);
    } finally {
      declined.destroy();
    }
    if (!SHADER_F16_AVAILABLE) {
      // (3') On an adapter that does not list it, true fails loudly (does not silently fall to f32)
      await assertRejects(() => acquireGpu({ shaderF16: true }), GpuFeatureError);
      return;
    }
    // (3) true adds the feature and returns only after **the canary actually runs** (a known answer of
    // RTNE with runtime values passed through an f16 shared tile). An implementation that returns on
    // enumeration alone would miss the silent all-zeros of denoland/deno#23125.
    const enabled = await acquireGpu({ shaderF16: true });
    try {
      assertEquals(enabled.shaderF16Enabled, true);
      assertEquals(enabled.features.has(SHADER_F16_FEATURE), true);
    } finally {
      enabled.destroy();
    }
  },
);

/**
 * A fake device with only the surface the canary touches (2 buffers, pipeline creation, 1 dispatch,
 * readback). The mismatch path and the loss path cannot be produced on a real GPU (the former needs
 * the environment of `denoland/deno#23125`, the latter a TDR during execution), so these 2 can be
 * pinned only here.
 */
const canaryDevice = (
  parts: {
    /** The f32 values observed by the readback. **If omitted, mapAsync never resolves** (an artificial window). */
    readonly mapped?: readonly number[];
    /** If true, popErrorScope does not resolve (an artificial window — see the comment of the test below). */
    readonly hangValidation?: boolean;
    readonly lost?: Promise<GPUDeviceLostInfo>;
  } = {},
): GPUDevice => {
  const buffer = {
    destroy: (): void => {},
    unmap: (): void => {},
    getMappedRange: (): ArrayBuffer => new Float32Array(parts.mapped ?? []).buffer,
    mapAsync: (): Promise<void> =>
      parts.mapped === undefined ? new Promise<void>(() => {}) : Promise.resolve(),
  };
  const pass = {
    setPipeline: (): void => {},
    setBindGroup: (): void => {},
    dispatchWorkgroups: (): void => {},
    end: (): void => {},
  };
  return {
    lost: parts.lost ?? new Promise<GPUDeviceLostInfo>(() => {}),
    createBuffer: () => buffer,
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => pass,
      copyBufferToBuffer: (): void => {},
      finish: () => ({}),
    }),
    pushErrorScope: (): void => {},
    popErrorScope: (): Promise<GPUError | null> =>
      parts.hangValidation === true
        ? new Promise<GPUError | null>(() => {})
        : Promise.resolve(null),
    queue: { submit: (): void => {} },
  } as unknown as GPUDevice;
};

/** The canary's known answer (the RTNE sum of a round trip through a shared f16 tile / f16 semantics of a constant expression). */
const CANARY_SOLUTION = [4.0029296875, 1] as const;

/** A loss that has already settled (`reason` is not used by the canary's branches, so a fixed value will do). */
const loss = (): Promise<GPUDeviceLostInfo> =>
  Promise.resolve(
    platformObject<GPUDeviceLostInfo>({ reason: "destroyed", message: "loss caused by the test" }),
  );

/**
 * A loss that settles in a macrotask. It makes the stages that advance by microtasks alone (the
 * errorScope pop of pipeline creation, etc.) always pass first, and pins the point of contention to
 * the "wait that never resolves" alone — with an immediately settled loss, who wins would be coupled
 * to the tick count of each stage, and the test would break merely because the number of pops
 * changed (it actually broke when the second scope was added).
 */
const lossAfterMacrotask = (): Promise<GPUDeviceLostInfo> =>
  new Promise((resolve) =>
    setTimeout(
      () =>
        resolve(
          platformObject<GPUDeviceLostInfo>({
            reason: "destroyed",
            message: "loss caused by the test",
          }),
        ),
      0,
    ),
  );

test("the shader-f16 canary passes only when it matches the known answer", async () => {
  await assertShaderF16Executes(canaryDevice({ mapped: CANARY_SOLUTION }));

  // The silent all-zeros of denoland/deno#23125, and an implementation in which no rounding happens (the shared tile passes values through)
  for (const observed of [
    [0, 0],
    [4.00146484375, 1],
  ]) {
    const error = await assertRejects(
      () => assertShaderF16Executes(canaryDevice({ mapped: observed })),
      GpuFeatureError,
    );
    assert(error.message.includes("does not match the known answer"), error.message);
  }
});

test("the shader-f16 canary fails on device loss instead of waiting forever (does not hang acquireGpu)", async () => {
  // This fake device **builds** "a wait that never resolves". In measurements (Deno + wgpu / NVIDIA,
  // 2026-08-16), popErrorScope resolves with null and mapAsync rejects even after a loss caused by
  // destroy, so this is not a reproduction of actual behaviour but the window of "an unverified
  // implementation (a real TDR / a browser) that returned no resolution". If even one wait that is
  // not raced remains, this await never returns in such an environment and acquireGpu hangs — this
  // test pins that down.
  const readback = assertShaderF16Executes(canaryDevice({ lost: lossAfterMacrotask() }));
  const readbackError = await assertRejects(() => readback, GpuDeviceLostError);
  assert(readbackError.message.includes("readback"), readbackError.message);

  const validation = assertShaderF16Executes(
    canaryDevice({ mapped: CANARY_SOLUTION, hangValidation: true, lost: loss() }),
  );
  const validationError = await assertRejects(() => validation, GpuDeviceLostError);
  assert(validationError.message.includes("pipeline creation"), validationError.message);
});

test("readAdapterInfo normalizes to safe empty values even for an adapter without info", () => {
  // Old Chromium (the requestAdapterInfo() era) has no adapter.info.
  const empty = readAdapterInfo({});
  assertEquals(empty.vendor, "");
  assertEquals(empty.architecture, "");
  assertEquals(empty.device, "");
  assertEquals(empty.description, "");
  assertEquals(empty.isFallbackAdapter, false);

  const present = { vendor: "acme", architecture: "a", device: "d", description: "x" };
  assertEquals(readAdapterInfo({ info: present as GPUAdapterInfo }).vendor, "acme");
});

test("withScopeLock serializes errorScope ranges per device", async () => {
  const gpu = fakeGpuContext(fakeDevice());
  const trace: string[] = [];
  let active = 0;
  let maxActive = 0;

  /** Uses the difference in `ticks` to build a form in which "a later caller could settle first" (without the lock the ranges would overlap). */
  const body = (label: string, ticks: number) => async (): Promise<string> => {
    active += 1;
    maxActive = Math.max(maxActive, active);
    trace.push(`${label}:enter`);
    for (let i = 0; i < ticks; i += 1) await Promise.resolve();
    trace.push(`${label}:exit`);
    active -= 1;
    return label;
  };

  const results = await Promise.all([
    gpu[RUNTIME_INTERNAL].withScopeLock(body("a", 6)),
    gpu[RUNTIME_INTERNAL].withScopeLock(body("b", 1)),
    gpu[RUNTIME_INTERNAL].withScopeLock(body("c", 3)),
  ]);

  assertEquals(maxActive, 1, "no 2 ranges run at the same time");
  assertEquals(trace, ["a:enter", "a:exit", "b:enter", "b:exit", "c:enter", "c:exit"]);
  assertEquals(results, ["a", "b", "c"], "each return value is the result of that call itself");
});

test("withScopeLock does not propagate one range's failure to the ones after it", async () => {
  const gpu = fakeGpuContext(fakeDevice());
  const failure = new Error("failure inside the range");

  const failed = gpu[RUNTIME_INTERNAL].withScopeLock(() => Promise.reject(failure));
  const following = gpu[RUNTIME_INTERNAL].withScopeLock(() => Promise.resolve("ok"));

  assertEquals(await assertRejects(() => failed), failure);
  assertEquals(
    await following,
    "ok",
    "the chain links only settlements, so the following call is not caught up in it",
  );
});

test("onLost can be unsubscribed, and notifies immediately if the device was already lost when subscribing", async () => {
  const { gpu, lose } = losableGpuContext();
  const seen: string[] = [];

  gpu.onLost(() => seen.push("kept"));
  const cancel = gpu.onLost(() => seen.push("cancelled"));
  cancel();

  lose();
  await gpu.device.lost;
  assertEquals(seen, ["kept"], "an unsubscribed listener is not called");

  // A subscription after the loss fires immediately (synchronously) so that nothing is missed
  gpu.onLost(() => seen.push("late"));
  assertEquals(seen, ["kept", "late"]);
});

test("raceDeviceLost turns a loss into an exception and unsubscribes on settlement", async () => {
  const { gpu, lose } = losableGpuContext();
  assertEquals(gpu.pendingLostListeners, 0);

  // If the work settles first, no subscription is left (if reactions piled up per flush / readback,
  // a long-lived Session would leak monotonically — this does not show in behaviour, so it is pinned by
  // the count)
  for (let i = 0; i < 3; i += 1) {
    assertEquals(await gpu[RUNTIME_INTERNAL].raceDeviceLost(Promise.resolve(7), "work"), 7);
  }
  assertEquals(gpu.pendingLostListeners, 0, "a settled wait leaves not a single subscription");

  // A wait that never resolves becomes an exception on loss (not a hang)
  const hanging = gpu[RUNTIME_INTERNAL].raceDeviceLost(new Promise<void>(() => {}), "flush");
  assertEquals(
    gpu.pendingLostListeners,
    1,
    "a subscription stands while waiting (the evidence that it is counted)",
  );
  lose();
  const error = await assertRejects(() => hanging, GpuDeviceLostError);
  assert(error.message.includes("flush"), error.message);
  assertEquals(gpu.pendingLostListeners, 0);
});

test("a device-loss exception carries the reason / message the backend set in its text", async () => {
  // The real cause (such as Metal's failure to allocate a counter sample buffer) remains only in
  // `GPUDeviceLostInfo.message` — errors of a lost device do not enter errorScope, so dropping it here
  // would leave the real cause nowhere in karume (the Metal timestamp section of docs/limitations.md).
  // NOTE: by the spec `reason` can also be "unknown", but Deno's type declarations have only
  // "destroyed". The point of the check is the message side (the real-cause string), so it is built
  // with a value the type accepts.
  const loss: GPUDeviceLostInfo = platformObject<GPUDeviceLostInfo>({
    reason: "destroyed",
    message: "Device::create_query_set: Not enough memory left.",
  });
  const { gpu, lose } = losableGpuContext();

  const hanging = gpu[RUNTIME_INTERNAL].raceDeviceLost(new Promise<void>(() => {}), "flush");
  lose(loss);
  const raced = await assertRejects(() => hanging, GpuDeviceLostError);
  assert(
    raced.message.startsWith("The device was lost during flush (rebuild required)"),
    `the prefix does not change: ${raced.message}`,
  );
  assert(raced.message.includes("reason: destroyed"), raced.message);
  assert(raced.message.includes("Device::create_query_set"), raced.message);

  // The synchronous check (assertDeviceUsable) carries the same reason. Even a fake device, once lost,
  // fails before reaching createBuffer, so no real GPU is needed.
  await gpu.device.lost;
  const gated = await assertRejects(() => gpu.createResident(16, "after-lost"), GpuDeviceLostError);
  assert(
    gated.message.startsWith("resident 'after-lost' allocation: device was lost"),
    `the prefix does not change: ${gated.message}`,
  );
  assert(gated.message.includes("Device::create_query_set"), gated.message);
});

test("loss notification is not stopped by a listener's exception, and does not swallow that exception either", async () => {
  const failure = new Error("failure of the user callback");
  // The public onDeviceLost is registered by the constructor, so it comes first in insertion order.
  // Without isolation, this throw would keep the notification from reaching the internal subscription
  // after it (the wait of raceDeviceLost), the await below would never return and the test would time
  // out — exactly the form in which a loss turns into a hang.
  const { gpu, lose } = losableGpuContext(() => {
    throw failure;
  });
  const hanging = gpu[RUNTIME_INTERNAL].raceDeviceLost(new Promise<void>(() => {}), "flush");
  assertEquals(
    gpu.pendingLostListeners,
    2,
    "the public notification and the internal subscription are both lined up",
  );

  // The isolated exception is not discarded but escapes asynchronously (keeps fail loudly, but
  // separated from the loss's control path). On Node it is observed as the process's
  // uncaughtException (test-support/process-events).
  const { reported: rethrown } = await captureUncaughtExceptions(async () => {
    lose();
    const error = await assertRejects(() => hanging, GpuDeviceLostError);
    assert(error.message.includes("flush"), error.message);
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  assertEquals(rethrown.length, 1);
  assertStrictEquals(rethrown[0], failure);
});

/**
 * A fake device with only the surface popFailureScopes touches (popErrorScope). `scopes`
 * corresponds to the order the pops are issued = [validation, out-of-memory] (the top of the stack
 * is validation).
 */
const poppingDevice = (scopes: readonly (string | undefined)[]): GPUDevice => {
  const queue = [...scopes];
  return {
    popErrorScope: (): Promise<GPUError | null> => {
      const message = queue.shift();
      return Promise.resolve(message === undefined ? null : ({ message } as GPUError));
    },
  } as unknown as GPUDevice;
};

test("popFailureScopes returns the OOM when both scopes capture something (the validation is derived)", async () => {
  // A createBuffer that runs out of room returns an invalid buffer, and a later operation that uses
  // it raises `Buffer with '' label is invalid` in validation. Returning the validation first would
  // discard the root-cause OOM and give a report indistinguishable from use after destroy (prevents a
  // measured incident from recurring).
  const both = await popFailureScopes(
    poppingDevice(["Buffer with '' label is invalid", "not enough memory left"]),
    "weight upload",
  );
  assertInstanceOf(
    both,
    GpuOutOfMemoryError,
    "returns the root-cause OOM, not the derived validation",
  );
  assert(both.message.startsWith("weight upload: "), both.message);
  assert(both.message.includes("not enough memory left"), both.message);

  // Pure validation (empty OOM scope) behaves as before
  const validationOnly = await popFailureScopes(
    poppingDevice(["Buffer with '' label has been destroyed", undefined]),
    "run encoding",
  );
  assertInstanceOf(validationOnly, GpuValidationError);
  assert(validationOnly.message.includes("has been destroyed"), validationOnly.message);

  assertEquals(await popFailureScopes(poppingDevice([]), "clean"), undefined);
});

test.skipIf(!GPU_AVAILABLE)(
  "a validation failure in concurrent errorScope ranges is attributed to the right side (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const { layout } = await new PipelineCache(gpu.device).get(
        "scope-lock-probe",
        `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x] + 1.0; }
}
`,
      );

      /**
       * A range of the same shape as #runOnce (push -> encoding across an await -> pop). The
       * difference in `ticks` builds an asymmetric overlap in which "the side that caused the error pops
       * first".
       *
       * Without the lock this shape is misattributed in measurements (faulty=clean / innocent=ERROR). pop
       * takes the top of the stack unconditionally, so faulty takes the empty scope of innocent pushed
       * after it, and innocent picks up faulty's error.
       */
      const scoped = (faulty: boolean, ticks: number): Promise<Error | undefined> =>
        gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
          pushFailureScopes(gpu.device);
          // A bindGroup missing binding 0 raises no synchronous exception and shows up only in errorScope
          if (faulty) gpu.device.createBindGroup({ layout, entries: [] });
          for (let i = 0; i < ticks; i += 1) await Promise.resolve();
          return await popFailureScopes(gpu.device, faulty ? "faulty" : "innocent");
        });

      const [faultyResult, innocentResult] = await Promise.all([scoped(true, 1), scoped(false, 8)]);

      assertInstanceOf(
        faultyResult,
        GpuValidationError,
        "the side that caused the error captures it",
      );
      assert(faultyResult.message.startsWith("faulty:"), faultyResult.message);
      assertEquals(
        innocentResult,
        undefined,
        "an unrelated range does not fail as collateral damage",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "GpuContext subscribes to device.lost and records it as distinct from an intentional destroy (real GPU)",
  async () => {
    let notified = 0;
    const gpu = await acquireGpu({ onDeviceLost: () => (notified += 1) });

    assertEquals(gpu.lost, undefined);
    assertEquals(gpu.destroyRequested, false);

    gpu.destroy();
    // The internal subscription was registered in the constructor, so it runs before this await
    const info = await gpu.device.lost;

    assertEquals(info.reason, "destroyed");
    assertEquals(gpu.destroyRequested, true);
    assertEquals(gpu.lost?.reason, "destroyed");
    assertEquals(notified, 0, "an intentional destroy is not a loss notification");
  },
);

test("subgroup32 requests itself and every feature it needs only when requested explicitly, and not by default", () => {
  const present = new Set(["subgroups", "subgroup-size-control"]),
    language = new Set(["subgroup_id"]);
  assertEquals(planSubgroup32Features(present, language, undefined), []);
  assertEquals(planSubgroup32Features(new Set(), new Set(), false), []);
  assertEquals(planSubgroup32Features(present, language, true), [
    "subgroups",
    "subgroup-size-control",
  ]);
  for (const absent of [
    new Set<string>(),
    new Set(["subgroups"]),
    new Set(["subgroup-size-control"]),
  ]) {
    assertThrows(() => planSubgroup32Features(absent, language, true), GpuFeatureError);
  }
  assertThrows(
    () => planSubgroup32Features(present, new Set(), true),
    GpuFeatureError,
    "subgroup_id",
  );
});

test("subgroups 'available' requests the features only when all three are present, and never rejects", () => {
  // ADR 0132 decision 5 (amends ADR 0100 decision 1): for bit-identical subgroup forms.
  const present = new Set(["subgroups", "subgroup-size-control"]);
  const language = new Set(["subgroup_id"]);
  assertEquals(planSubgroup32Features(present, language, "available"), [
    "subgroups",
    "subgroup-size-control",
  ]);
  for (const [features, wgsl] of [
    [new Set<string>(), language],
    [new Set(["subgroups"]), language],
    [new Set(["subgroup-size-control"]), language],
    [present, new Set<string>()],
  ] as const) {
    assertEquals(planSubgroup32Features(features, wgsl, "available"), []);
  }
  // Anything but a boolean or "available" still throws (a JS caller outside the type).
  const jsCaller = planSubgroup32Features as unknown as (
    features: ReadonlySet<string>,
    language: ReadonlySet<string>,
    requested: unknown,
  ) => readonly string[];
  for (const requested of ["true", "required", 1]) {
    assertThrows(() => jsCaller(present, language, requested), GpuFeatureError, "subgroups must");
  }
});

test("the subgroup lane-mapping probe is true only when every lane reports the linear mapping, and does not wait on a lost device", async () => {
  assertEquals(
    await probeSubgroupLaneMapping(canaryDevice({ mapped: Array<number>(256).fill(1) })),
    true,
  );
  for (const mapped of [[], Array<number>(256).fill(0), [...Array<number>(255).fill(1), 0]]) {
    assertEquals(await probeSubgroupLaneMapping(canaryDevice({ mapped })), false);
  }
  await assertRejects(
    () => probeSubgroupLaneMapping(canaryDevice({ lost: lossAfterMacrotask() })),
    GpuDeviceLostError,
    "subgroups",
  );
});

test("the subgroup32 canary accepts only the known answer on every lane and does not wait forever on loss", async () => {
  await assertSubgroup32Executes(canaryDevice({ mapped: Array<number>(256).fill(32896) }));
  for (const mapped of [[], Array<number>(256).fill(0), [...Array<number>(255).fill(32896), 0]]) {
    await assertRejects(() => assertSubgroup32Executes(canaryDevice({ mapped })), GpuFeatureError);
  }
  await assertRejects(
    () => assertSubgroup32Executes(canaryDevice({ lost: lossAfterMacrotask() })),
    GpuDeviceLostError,
    "subgroups",
  );
  await assertRejects(
    () =>
      assertSubgroup32Executes(canaryDevice({ hangValidation: true, lost: lossAfterMacrotask() })),
    GpuDeviceLostError,
    "subgroups",
  );
});

test("batch finish: a range whose device was lost while waiting for the errorScope pop does not return success", async () => {
  // A finish without readback races only `onSubmittedWorkDone` against the loss and does not race the
  // pop wait after it (a pop after the loss settles with null — docs/research/2026-08-16-device-lost-wait-settlement.md).
  // A finish across a loss that returned success would be misread as "the range closed safely", so
  // the loss is checked after the pop.
  const pops: PromiseWithResolvers<GPUError | null>[] = [];
  const { gpu, lose } = losableGpuContext(undefined, {
    popErrorScope: () => {
      const pop = Promise.withResolvers<GPUError | null>();
      pops.push(pop);
      return pop.promise;
    },
  });
  const batch = await gpu.beginBatch();
  const finished = batch.finish();
  // Wait until the pop is issued (= the fence has been passed).
  while (pops.length < 2) await Promise.resolve();
  lose();
  // Build the form in which recording the loss (the then of `device.lost`) runs before the pop settles.
  for (let i = 0; i < 4; i += 1) await Promise.resolve();
  for (const pop of pops) pop.resolve(null);
  await assertRejects(() => finished, GpuDeviceLostError, "batch completion");
});

test("createResident: on a device lost while waiting for the errorScope pop, does not return success and destroys the allocated buffer", async () => {
  // After passing the synchronous check at the entry, a loss during the pop wait makes the pop settle
  // with null (docs/research/2026-08-16-device-lost-wait-settlement.md). Returning success here would
  // circulate an unusable resident tensor as a successful allocation, and the failure point would move
  // to the first write / read.
  const pops: PromiseWithResolvers<GPUError | null>[] = [];
  let destroyed = 0;
  const { gpu, lose } = losableGpuContext(undefined, {
    popErrorScope: () => {
      const pop = Promise.withResolvers<GPUError | null>();
      pops.push(pop);
      return pop.promise;
    },
    createBuffer: () =>
      ({
        destroy: (): void => {
          destroyed += 1;
        },
      }) as unknown as GPUBuffer,
  });
  const created = gpu.createResident(16, "racing");
  assertEquals(pops.length, 2, "2 pops are issued after the allocation");
  lose();
  // Build the form in which recording the loss (the then of `device.lost`) runs before the pop settles.
  for (let i = 0; i < 4; i += 1) await Promise.resolve();
  for (const pop of pops) pop.resolve(null);
  await assertRejects(() => created, GpuDeviceLostError, "resident 'racing' allocation");
  assertEquals(destroyed, 1, "the allocated buffer is dropped without being destroyed");
});

test("beginBatch does not open on a device whose destroy was requested or that was lost, and takes no range lock", async () => {
  // If it opened, it would keep accepting enqueues while holding the range lock, and the failure would
  // be delayed until finish. `destroy()` sets the flag synchronously, so it fails without waiting for
  // the reaction of `device.lost`.
  const destroyed = fakeGpuContext(fakeDevice());
  destroyed.destroy();
  await assertRejects(() => destroyed.beginBatch(), GpuDeviceLostError, "batch start");
  assertEquals(
    await destroyed[RUNTIME_INTERNAL].withScopeLock(() => Promise.resolve("next")),
    "next",
    "the rejected beginBatch took no lock (later ranges are not kept waiting)",
  );

  const { gpu, lose } = losableGpuContext();
  lose();
  await gpu.device.lost;
  await assertRejects(() => gpu.beginBatch(), GpuDeviceLostError, "batch start");
});

test("beginBatch: on a device lost while waiting for the lock, does not open the range and returns the lock at once", async () => {
  // The synchronous check at call time passes, but the device is lost while waiting for an earlier
  // range to return the lock. If it opened, it would keep holding the range lock on a lost device, and
  // later ranges would wait until the caller finishes.
  const { gpu, lose } = losableGpuContext();
  const first = await gpu.beginBatch();
  const second = gpu.beginBatch();
  lose();
  await gpu.device.lost;
  await first.finish().catch(() => undefined);
  await assertRejects(() => second, GpuDeviceLostError, "batch start");
  assertEquals(
    await gpu[RUNTIME_INTERNAL].withScopeLock(() => Promise.resolve("next")),
    "next",
    "the rejected beginBatch still holds the lock (later ranges are kept waiting)",
  );
});
