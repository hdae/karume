// Helper that wraps a fake device in a **real** GpuContext.
//
// MUST: GpuContext itself is never faked. What the tests check is GpuContext's production code —
// the errorScope section lock, the device-loss subscription and the like — and replacing it would
// leave the tests checking only the fake. Replace just the GPU itself (the device).

import { platformObject } from "@karume/test-support/webgpu";
import {
  type DeviceLostHandler,
  GpuContext,
  readAdapterInfo,
  type RequiredLimits,
} from "../../src/gpu/device.ts";

/**
 * The lock and the loss subscription never look at the limits, so every requested key at 0 is
 * enough.
 */
const ZERO_LIMITS: RequiredLimits = {
  maxBufferSize: 0,
  maxStorageBufferBindingSize: 0,
  maxUniformBufferBindingSize: 0,
  maxStorageBuffersPerShaderStage: 0,
  maxUniformBuffersPerShaderStage: 0,
  maxComputeWorkgroupStorageSize: 0,
  maxComputeInvocationsPerWorkgroup: 0,
  maxComputeWorkgroupSizeX: 0,
  maxComputeWorkgroupSizeY: 0,
  maxComputeWorkgroupSizeZ: 0,
  maxComputeWorkgroupsPerDimension: 0,
};

/**
 * The smallest fake device: only the surface GpuContext touches at construction (features /
 * lost).
 */
export type FakeDeviceParts = {
  /** Absent = "a device that is never lost" (a promise that never settles). */
  readonly lost?: Promise<GPUDeviceLostInfo>;
  readonly features?: Iterable<string>;
  /**
   * The errorScope pop (needed by checks that go through a batch's settlement). Absent = settles
   * at once with null.
   */
  readonly popErrorScope?: () => Promise<GPUError | null>;
  /** `queue.onSubmittedWorkDone` (a batch's completion fence). Absent = settles at once. */
  readonly onSubmittedWorkDone?: () => Promise<void>;
  /** `queue.submit` (needed by checks that observe that it was called). Absent = does nothing. */
  readonly submit?: (commandBuffers: readonly GPUCommandBuffer[]) => void;
  /**
   * `createBuffer` (needed by checks that observe the destruction of allocated buffers). Absent =
   * the device has no such method.
   */
  readonly createBuffer?: (descriptor: GPUBufferDescriptor) => GPUBuffer;
  /**
   * `queue.writeBuffer` (needed by checks that observe what a GenerationContext writes to its
   * lengths uniform). Absent = the queue has no such method.
   */
  readonly writeBuffer?: (buffer: GPUBuffer, offset: number, data: BufferSource) => void;
};

/** Passed through a cast because the whole DOM type is not reproduced (a test-only boundary). */
export const fakeDevice = (parts: FakeDeviceParts = {}): GPUDevice =>
  ({
    lost: parts.lost ?? new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set(parts.features ?? []),
    // Needed by checks that go through `GpuContext.destroy()` (the synchronous destroy-request
    // flag).
    destroy: (): void => undefined,
    pushErrorScope: (): void => undefined,
    popErrorScope: parts.popErrorScope ?? ((): Promise<GPUError | null> => Promise.resolve(null)),
    createBuffer: parts.createBuffer,
    queue: {
      onSubmittedWorkDone: parts.onSubmittedWorkDone ?? ((): Promise<void> => Promise.resolve()),
      submit: parts.submit ?? ((): void => undefined),
      ...(parts.writeBuffer === undefined ? {} : { writeBuffer: parts.writeBuffer }),
    },
  }) as unknown as GPUDevice;

export const fakeGpuContext = (device: GPUDevice, onDeviceLost?: DeviceLostHandler): GpuContext =>
  new GpuContext(device, readAdapterInfo({}), ZERO_LIMITS, new Set(), onDeviceLost);

/** The loss info when the trigger is given nothing (GpuContext does not branch on `reason`). */
const DEFAULT_LOSS: GPUDeviceLostInfo = platformObject<GPUDeviceLostInfo>({
  reason: "destroyed",
  message: "lost by the test",
});

/**
 * A fake device whose loss can be triggered later, and its trigger.
 *
 * `lose` takes the info so a test can pin that the real cause the backend supplies (`reason` /
 * `message`) is carried into the exception's message.
 *
 * `onDeviceLost` can be passed to reproduce the shape where the public notification comes **first
 * in insertion order** (registered in the constructor). Being called before the internal
 * subscriptions is exactly the condition that requires the exception isolation.
 */
export const losableGpuContext = (
  onDeviceLost?: DeviceLostHandler,
  parts: Omit<FakeDeviceParts, "lost"> = {},
): {
  readonly gpu: GpuContext;
  readonly lose: (info?: GPUDeviceLostInfo) => void;
} => {
  let resolve: (info: GPUDeviceLostInfo) => void = () => {};
  const lost = new Promise<GPUDeviceLostInfo>((settle) => {
    resolve = settle;
  });
  return {
    gpu: fakeGpuContext(fakeDevice({ ...parts, lost }), onDeviceLost),
    lose: (info: GPUDeviceLostInfo = DEFAULT_LOSS): void => resolve(info),
  };
};

/**
 * A fake GPU on which a Session derives its recipes: it compiles any WGSL to a pipeline that does
 * nothing and hands out buffers that hold nothing. Its limits are {@link fakeGpuContext}'s zeros but
 * the WebGPU default workgroup count per dimension (65535, which a recipe's dispatch counts are
 * checked against), so a run stops at the transient plan (`invalid transient plan limit`), after
 * every step was planned and every recipe built — enough to observe a refusal raised while either
 * happens, without a GPU. The device parts go through a cast, as in {@link fakeDevice}.
 */
export const fakeRecipeGpuContext = (): GpuContext => {
  const device = fakeDevice({
    createBuffer: (descriptor) =>
      ({
        size: descriptor.size,
        usage: descriptor.usage,
        label: descriptor.label ?? "",
        destroy: (): void => undefined,
      }) as unknown as GPUBuffer,
    writeBuffer: (): void => undefined,
  });
  return new GpuContext(
    Object.assign(device, {
      limits: { minStorageBufferOffsetAlignment: 256 },
      createShaderModule: () => ({
        getCompilationInfo: () => Promise.resolve({ messages: [] }),
      }),
      createComputePipelineAsync: () => Promise.resolve({ getBindGroupLayout: () => ({}) }),
    }),
    readAdapterInfo({}),
    { ...ZERO_LIMITS, maxComputeWorkgroupsPerDimension: 65535 },
    new Set(),
  );
};
