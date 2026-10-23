/**
 * Acquiring the WebGPU device and normalizing its capabilities.
 *
 * The job of this layer: "normalize the surfaces that differ between implementations (features /
 * limits / WGSL language features) in one place, and turn every silent failure into a loud
 * exception".
 *
 * MUST: do not capture the device at module scope. Continuing after the device is destroyed means
 * rebuilding from `acquireGpu()` (an individual `buffer.destroy()` returns that buffer's share at
 * the next poll, but the layers that hold the device are bound to the device's lifetime). Every
 * layer that holds the device (PipelineCache / SubmitScheduler / RunArena) has to be structured so
 * it can be rebuilt with the same lifetime as GpuContext.
 *
 * **This is the layer's entry point**; the implementation is split in two: {@link "./acquire.ts"}
 * (everything up to taking the device and assembling a GpuContext — adapter acquisition, planning
 * and checking limits / features, running the canaries) and {@link "./context.ts"} (the container
 * of an acquired device — GpuContext / ResidentTensor / BatchScope and the runtime-internal
 * surface). This file only re-exports the public names of both and holds no implementation.
 *
 * MUST: keep the value (runtime) dependency **one-way, acquire → context**. Only `acquireGpu`
 * constructs a GpuContext, so no value import is needed in the other direction; adding one would
 * make device acquisition / the canaries and the container's lifetime management circular (context
 * refers to acquire only through `import type` — it is erased, so the runtime import graph stays
 * one-way).
 * MUST: users, other runtime modules and tests **import through this name (`device.ts`)**. It is a
 * facade that keeps a single entry point for the layer; the split into two files is only an
 * implementation convenience.
 */

export {
  acquireGpu,
  assertLimitsGranted,
  assertShaderF16Executes,
  assertSubgroup32Executes,
  probeSubgroupLaneMapping,
  GpuFeatureError,
  GpuLimitError,
  GpuUnavailableError,
  LIMIT_CAPS,
  planRequiredLimits,
  planShaderF16Feature,
  planSubgroup32Features,
  planTimestampFeature,
  readAdapterInfo,
  readAdapterLimits,
  REQUIRED_LIMIT_KEYS,
} from "./acquire.ts";
export type { AcquireGpuOptions, DeviceLostHandler, RequiredLimits } from "./acquire.ts";
export {
  assertDeviceUsable,
  BatchScope,
  BatchScopeError,
  describeDeviceLoss,
  GpuContext,
  GpuDeviceLostError,
  ResidentTensor,
  ResidentTensorError,
  RUNTIME_INTERNAL,
  SHADER_F16_FEATURE,
} from "./context.ts";
export type { BatchFailureOrigin, BatchMember, BatchReadSource, ResidentData } from "./context.ts";
