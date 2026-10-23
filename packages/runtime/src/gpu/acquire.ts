/**
 * Everything up to taking the device and assembling a {@link GpuContext} (adapter acquisition →
 * planning and checking limits → planning features and running the canaries → normalizing
 * adapterInfo / WGSL language features).
 *
 * MUST: the dependency is one-way, acquire → context. Only {@link acquireGpu} in this file
 * constructs a {@link GpuContext}, and the container side (`context.ts`) does not value-import from
 * here. The entry point of the layer is `device.ts`, which re-exports the public names of both
 * files.
 */

import {
  describeDeviceLoss,
  GpuContext,
  GpuDeviceLostError,
  SHADER_F16_FEATURE,
  TIMESTAMP_QUERY_FEATURE,
} from "./context.ts";
import { withPipelineScope } from "./error-scope.ts";
import { CodegenError } from "../codegen/errors.ts";
import {
  assertGeometryProfile,
  assertGeometryProfileWithinLimits,
  type GeometryProfile,
} from "../kernels/geometry-profile.ts";
import { BUFFER_USAGE, MAP_MODE } from "./webgpu-constants.ts";

/** navigator.gpu is missing / no adapter can be obtained. */
export class GpuUnavailableError extends Error {
  override readonly name = "GpuUnavailableError";
}

/**
 * The device does not meet a requested limit (detects a missed request or a demotion to the spec
 * default).
 */
export class GpuLimitError extends Error {
  override readonly name = "GpuLimitError";
}

/**
 * The adapter lacks a feature specified as required ({@link AcquireGpuOptions.gpuTiming} = `true`
 * and the like). It is separate from a limit shortfall ({@link GpuLimitError}) because the user
 * branches differently — with a feature one can continue by "giving up and dropping the feature in
 * that environment", but with a limit shortfall one must not continue.
 */
export class GpuFeatureError extends Error {
  override readonly name = "GpuFeatureError";
}

/**
 * The limits requested explicitly in requiredLimits.
 *
 * MUST: a limit that is not requested falls to the **specification default**, not the adapter
 * value. The defaults are 16384 bytes of workgroup storage / 256 invocations per workgroup /
 * workgroupSize (256,256,64); without a request, kernels top out there even if the adapter could
 * offer 1024 invocations.
 *
 * MUST: `maxComputeWorkgroupSizeX/Y/Z` are limits separate from
 * `maxComputeInvocationsPerWorkgroup`. Raising only invocations does not let `@workgroup_size(512)`
 * through, and **the failure is silent** (`createComputePipeline` does not throw; dispatches to the
 * invalid pipeline become no-ops and the output is all 0). This silent failure can be made visible
 * only with {@link withPipelineScope}.
 */
export const REQUIRED_LIMIT_KEYS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
] as const;

type RequiredLimitKey = (typeof REQUIRED_LIMIT_KEYS)[number];

export type RequiredLimits = Readonly<Record<RequiredLimitKey, number>>;

/**
 * A setting that **lowers the upper bounds** of requiredLimits (takes `min` per key).
 *
 * MUST: usable only in the lowering direction (a request above the adapter value stays at the
 * adapter value). If it could be used for raising, a path where requestDevice fails as a whole
 * would silently appear.
 */
type LimitCaps = Partial<Record<RequiredLimitKey, number>>;

/**
 * Builds the requiredLimits passed to requestDevice from the adapter's real values.
 * When the requested values are the adapter values themselves, requestDevice always succeeds.
 *
 * MUST: per the specification `maxStorageBufferBindingSize ≤ maxBufferSize`. Raising only one of
 * them makes requestDevice fail as a whole, so the two are always planned together.
 *
 * `caps` are **test-only reductions** ({@link LIMIT_CAPS}). They only lower requested values, but a
 * cap on `maxBufferSize` moves the very value `maxStorageBufferBindingSize` is clamped to.
 * MUST: apply the cap to `maxBufferSize` **first** and clamp the binding limit with that value.
 * Applied afterwards, caps that lower only `maxBufferSize` would yield a plan with
 * `maxStorageBufferBindingSize > maxBufferSize`, and the purpose of this surface, "reproducing a
 * machine with core defaults", would silently be missed (the buffer limit drops while the binding
 * limit is not lowered). The remaining keys are independent of each other, so the order of applying
 * caps does not matter for them.
 */
export const planRequiredLimits = (
  adapterLimits: GPUSupportedLimits,
  caps: LimitCaps = {},
): RequiredLimits => {
  const maxBufferSize = Math.min(adapterLimits.maxBufferSize, caps.maxBufferSize ?? Infinity);
  // The exhaustiveness check of Record<RequiredLimitKey, number> keeps this in sync with the key
  // list by type.
  const planned: Record<RequiredLimitKey, number> = {
    maxBufferSize,
    maxStorageBufferBindingSize: Math.min(adapterLimits.maxStorageBufferBindingSize, maxBufferSize),
    maxUniformBufferBindingSize: adapterLimits.maxUniformBufferBindingSize,
    maxStorageBuffersPerShaderStage: adapterLimits.maxStorageBuffersPerShaderStage,
    maxUniformBuffersPerShaderStage: adapterLimits.maxUniformBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupsPerDimension: adapterLimits.maxComputeWorkgroupsPerDimension,
  };
  for (const key of REQUIRED_LIMIT_KEYS) {
    const cap = caps[key];
    if (cap !== undefined) planned[key] = Math.min(planned[key], cap);
  }
  return planned;
};

/**
 * Checks that the acquired device's limits are at least the requested ones.
 * A missed request or an implementation-driven demotion shows up as "kernels silently top out", so
 * it fails here.
 */
export const assertLimitsGranted = (planned: RequiredLimits, granted: GPUSupportedLimits): void => {
  const shortfalls = REQUIRED_LIMIT_KEYS.filter((key) => granted[key] < planned[key]).map(
    (key) => `${key}: granted ${granted[key]} < required ${planned[key]}`,
  );
  if (shortfalls.length > 0) {
    throw new GpuLimitError(`requiredLimits are not satisfied: ${shortfalls.join(", ")}`);
  }
};

/**
 * MUST NOT: request features that are not used. A requested feature not only widens the device's
 * capability surface but, depending on the adapter's implementation, can also make acquisition
 * fail. The weight storage paths do not emit `enable f16` for f32 / f16 / i8 (unpack2x16float — ADR
 * 0018), so **no feature is requested unconditionally**.
 * NOTE: ADR 0021 reread the "request nothing" policy as "request nothing **by default**". The
 * features added conditionally are GPU-side time measurement ({@link TIMESTAMP_QUERY_FEATURE} — the
 * three values of {@link AcquireGpuOptions.gpuTiming}), f16 **compute** variants
 * ({@link SHADER_F16_FEATURE} — {@link AcquireGpuOptions.shaderF16}) and 32-lane subgroups (ADR
 * 0100). The enabled features can be queried with {@link GpuContext.features}.
 */
const REQUIRED_FEATURES: readonly GPUFeatureName[] = [];

/**
 * The shape that takes out only the surface read as a feature set. Needed to accept **both**
 * `adapter.features` (`ReadonlySet<GPUFeatureName>`) and a plain `Set<string>` (the former has a
 * narrow element type, the latter does not satisfy `ReadonlySet<GPUFeatureName>`).
 */
type GpuFeatureSet = Iterable<string> & { has(feature: string): boolean };

/**
 * Decides "request timestamp-query or not" from the three values of
 * {@link AcquireGpuOptions.gpuTiming}.
 *
 * MUST: `undefined` means "do not request" (the same discipline as {@link planShaderF16Feature}).
 * No automatic decision — a device with measurement enabled opens 1 dispatch = 1 pass, so **wall
 * time would change with the adapter's capabilities** (measured: 370 to 375 ms per step,
 * independent of resolution, and the unit cost grows with the cumulative number of GPUBuffers
 * created; `docs/research/2026-08-04-host-overhead-recon.md` §3.2/§4.2). Handing out diagnostics
 * for free would make "measuring slows things down" the default and distort the perf baseline
 * itself.
 *
 * It is a pure function cut out of device acquisition because the branch `true` × feature missing
 * **cannot be produced on real hardware** (a feature the adapter has cannot be removed). Only the
 * decision is shaped to be unit-testable.
 */
export const planTimestampFeature = (
  adapterFeatures: GpuFeatureSet,
  requested: boolean | undefined,
): boolean => {
  if (requested !== true) return false;
  if (!adapterFeatures.has(TIMESTAMP_QUERY_FEATURE)) {
    throw new GpuFeatureError(
      `gpuTiming: true was specified, but the adapter lacks '${TIMESTAMP_QUERY_FEATURE}'` +
        ` (available: ${[...adapterFeatures].sort().join(", ") || "none"}). ` +
        "To continue without timing, omit gpuTiming (= do not request it) or set it to false",
    );
  }
  return true;
};

/**
 * Decides "request shader-f16 or not" from {@link AcquireGpuOptions.shaderF16}.
 *
 * MUST: `undefined` means "do not request" (the same discipline as {@link planTimestampFeature}).
 * f16 compute variants **change the numbers**, so if the adapter's capabilities decided whether
 * they are enabled, "change the machine and the output silently changes" would follow. Nothing but
 * opt-in is possible (ADR 0028).
 *
 * Like {@link planTimestampFeature} it is cut out as a pure function because the branch `true` ×
 * feature missing **cannot be produced on real hardware** (a feature the adapter has cannot be
 * removed).
 */
export const planShaderF16Feature = (
  adapterFeatures: GpuFeatureSet,
  requested: boolean | undefined,
): boolean => {
  if (requested !== true) return false;
  if (!adapterFeatures.has(SHADER_F16_FEATURE)) {
    throw new GpuFeatureError(
      `shaderF16: true was specified, but the adapter lacks '${SHADER_F16_FEATURE}'` +
        ` (available: ${[...adapterFeatures].sort().join(", ") || "none"}). ` +
        "The f16 compute variants ('f16' of attentionCompute / linearCompute) cannot be used in this environment — " +
        "omit shaderF16 to run on the default f32 path",
    );
  }
  return true;
};

/**
 * On a device where shader-f16 was requested and enabled, **runs a tiny f16 kernel with a known
 * answer** and compares (1 dispatch + readback).
 *
 * MUST: do not take the list of features as evidence that it "works". `denoland/deno#23125` (open)
 * reports that **a compute shader with `enable f16` on a device that requested shader-f16 outputs
 * all 0 without any error or warning**, while the same code works on another implementation.
 * Trusting the list alone would give a silent failure of the form "f16 variants were chosen, and
 * every attention / linear returns 0" — the same thinking as Karume keeping an errorScope always on
 * for "the silent no-op of an invalid pipeline"; here there is no way to detect it other than a
 * check that actually runs.
 *
 * Two checks:
 *
 * 1. `out[0] == 4.0029296875` — a round trip of **run-time values** through an f16 shared tile. The
 *    non-constant derived from `lid` is not removed by constant folding, and the rounding is
 *    enforced by the **element type** of shared memory. The 4 lanes' values are `1 + lid·2^-11`;
 *    with binary16 RTNE they become
 *    `1.0 / 1.0 (tie → even) / 1+2^-10 / 1+2^-9 (tie → even)` = total 4.0029296875.
 *    All 0 (denoland/deno#23125), an implementation that does not round (total 4.00146484375), and
 *    an implementation that bypasses shared memory all turn this one red.
 * 2. `out[1] == 1.0` — f16 semantics in a constant expression (the tie of `1 + 2^-11` goes to the
 *    even side, 1.0). This most likely looks at **compile-time folding**, so it is treated as an
 *    aid to ①.
 *
 * NOTE (measured in this phase, 2026-08-04): a round trip in registers, `vec4<f32>(vec4<f16>(x))`,
 * is **removed as an identity** by this environment's compiler (confirmed by fault injection).
 * Rounding is guaranteed only by the form "going through f16 **storage**", and the main kernels
 * rely on this property by rounding through the element type of the shared tile. The canary goes
 * through a shared tile for the same reason.
 *
 * The cost is paid only by calls that pass `shaderF16: true`.
 */
const SHADER_F16_CANARY_WGSL = `// karume shader-f16 canary (compare a known answer — a listed feature is no evidence that it "works")
enable f16;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

var<workgroup> tile: array<f16, 4>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  // Pass run-time values (derived from lid) through the f16 shared tile — a path constant folding
  // cannot remove
  tile[lid.x] = f16(1.0 + f32(lid.x) * 0.00048828125);
  workgroupBarrier();
  if (lid.x == 0u) {
    // ① round trip through the shared f16 tile + RTNE (1.0 + 1.0 + 1.0009765625 + 1.001953125)
    var acc = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
      acc = acc + f32(tile[i]);
    }
    out[0] = acc;
    // ② f16 semantics in a constant expression (1 + 2^-11 is a binary16 tie → the even side, 1.0)
    out[1] = f32(f16(1.0 + 0.00048828125));
  }
}
`;

/**
 * The canary's expected values (① RTNE through the shared f16 tile / ② f16 semantics of a constant
 * expression).
 */
const SHADER_F16_CANARY_EXPECTED: readonly number[] = [4.0029296875, 1];

/**
 * A device-loss race **local** to the canary section. Races the settling of `work` against
 * `device.lost`, and if the loss comes first, turns it into {@link GpuDeviceLostError}.
 *
 * MUST: every wait of the canary goes through here without exception. The ground, however, is not
 * that "a wait after a loss always hangs" — measured (Deno + wgpu / NVIDIA, 2026-08-16), after a
 * loss caused by `device.destroy()`, `popErrorScope` **resolves with null** (the same with an empty
 * stack), `mapAsync` rejects with `OperationError`, and every wait issued before the loss and in
 * flight also settled (the same side as the measurements recorded in
 * `docs/research/2026-08-01-m0-review.md`). We keep racing because **losses other than destroy (a
 * real TDR / driver reset / browser implementations) are unverified**; this is insurance so that
 * `acquireGpu()` does not hang and never return when it meets an implementation that does not
 * resolve.
 *
 * MUST NOT: carry this shape into layers with many waits. `device.lost.then(...)` is a reaction
 * with no way to remove it, so attaching it to repeated waits such as flush / readback makes it
 * grow monotonically (a single subscription is the responsibility of {@link GpuContext.onLost}).
 * The canary runs only once **before** the `GpuContext` is created and adds only a few per
 * acquisition, so here alone a direct race is allowed.
 */
const raceCanaryDeviceLost = <T>(
  device: GPUDevice,
  work: Promise<T>,
  where: string,
  feature = "shader-f16",
): Promise<T> =>
  Promise.race([
    work,
    device.lost.then((info): never => {
      throw new GpuDeviceLostError(
        `${feature} canary: device was lost during ${where} (rebuild required)${describeDeviceLoss(
          info,
        )}`,
      );
    }),
  ]);

/**
 * The canary itself (runs {@link SHADER_F16_CANARY_WGSL} and compares).
 *
 * NOTE: exported for tests (the mismatch path and the loss path cannot be produced on a real GPU).
 * The public surface is the explicit list in mod.ts, so exporting here does not widen the API
 * surface (ADR 0008).
 */
export const assertShaderF16Executes = async (device: GPUDevice): Promise<void> => {
  const byteLength = SHADER_F16_CANARY_EXPECTED.length * 4;
  const out = device.createBuffer({
    size: byteLength,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: byteLength,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  try {
    const pipeline = await raceCanaryDeviceLost(
      device,
      withPipelineScope(device, "shader-f16 canary", () =>
        device.createComputePipeline({
          layout: "auto",
          compute: {
            module: device.createShaderModule({ code: SHADER_F16_CANARY_WGSL }),
            entryPoint: "main",
          },
        }),
      ),
      "pipeline creation",
    );
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: out } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(out, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await raceCanaryDeviceLost(device, staging.mapAsync(MAP_MODE.READ), "readback");
    const observed = [...new Float32Array(staging.getMappedRange().slice(0))];
    staging.unmap();
    const matches = SHADER_F16_CANARY_EXPECTED.every((value, index) => observed[index] === value);
    if (!matches) {
      throw new GpuFeatureError(
        `'${SHADER_F16_FEATURE}' is listed, but an actual run of the f16 kernel does not match the known answer` +
          ` (expected [${SHADER_F16_CANARY_EXPECTED.join(", ")}] / observed [${observed.join(", ")}]). ` +
          "The f16 implementation of this environment cannot be trusted (all zeros is the same silent failure as denoland/deno#23125; " +
          "a mismatch in the first element alone means the rounding through the shared f16 tile did not happen). Omit shaderF16 and " +
          "run on the default f32 path",
      );
    }
  } finally {
    staging.destroy();
    out.destroy();
  }
};

/**
 * MUST: `adapter.info` is a surface that differs between implementations. The specification makes
 * it a required property, but old Chromium from the `requestAdapterInfo()` era does not have it and
 * it is undefined. Since the type declares it required, the difference can only be absorbed on the
 * reading side, so access is confined to this one function and normalized to empty values (this
 * guarantees that {@link GpuContext.adapterInfo} is always safe to read).
 */
type AdapterInfoHost = { readonly info?: GPUAdapterInfo };

/**
 * adapter information normalized to empty values (one object shared by every
 * {@link readAdapterInfo}).
 * MUST: freeze it — it is a shared value that reaches user callbacks (the function form of
 * `geometryProfile`) and `GpuContext.adapterInfo`, and freezing keeps a rewrite in one place from
 * leaking into the adapter information of every later device (the material for automatic selection
 * and matching) (a real `GPUAdapterInfo` has read-only attributes too, and cannot be rewritten
 * either).
 */
// `@webgpu/types` brands the platform interface (`__brand`); the literal carries every observable
// field, so the cast only supplies the brand.
const EMPTY_ADAPTER_INFO: GPUAdapterInfo = Object.freeze<Omit<GPUAdapterInfo, "__brand">>({
  vendor: "",
  architecture: "",
  device: "",
  description: "",
  subgroupMinSize: 0,
  subgroupMaxSize: 0,
  isFallbackAdapter: false,
}) as GPUAdapterInfo;

export const readAdapterInfo = (adapter: AdapterInfoHost): GPUAdapterInfo =>
  adapter.info ?? EMPTY_ADAPTER_INFO;

/**
 * MUST: `navigator.gpu.wgslLanguageFeatures` is a surface that differs between implementations
 * (some implementations lack it in their type definitions, some do not provide it). Direct access
 * is confined to this one function, and when it is missing it degrades to an empty set.
 * On its own it is not proof of GPU support (it is listed regardless of hardware support).
 * What it may be used for in automatic selection is **choosing between numerically identical
 * variants** (dot4I8Packed / emulation of w8a8 — ADR 0025. Only where a wrong choice does not
 * change the result by a single bit may it drive a speed branch).
 * subgroup_id is checked only as a necessary condition when explicitly requested, and the device
 * feature and a canary run are required too.
 */
type WgslLanguageFeatureHost = GPU & { readonly wgslLanguageFeatures?: Iterable<string> };

const readWgslLanguageFeatures = (gpu: GPU): ReadonlySet<string> => {
  const host: WgslLanguageFeatureHost = gpu;
  const features = host.wgslLanguageFeatures;
  return new Set(features ?? []);
};

/**
 * Checks the 3 capabilities needed for execution only when requested. `true` = required (a
 * shortfall is rejected — no silent fallback); `"available"` = requested when all three are present,
 * otherwise nothing is requested and nothing is rejected (ADR 0132 decision 5 amends ADR 0100
 * decision 1 — for code whose subgroup form is bit-identical to its portable form).
 */
export const planSubgroup32Features = (
  adapterFeatures: GpuFeatureSet,
  languageFeatures: ReadonlySet<string>,
  requested: boolean | "available" | undefined,
): readonly string[] => {
  if (requested !== undefined && typeof requested !== "boolean" && requested !== "available") {
    throw new GpuFeatureError("subgroups must be a boolean or 'available'");
  }
  if (requested === "available") {
    return adapterFeatures.has("subgroups") &&
      adapterFeatures.has("subgroup-size-control") &&
      languageFeatures.has("subgroup_id")
      ? ["subgroups", "subgroup-size-control"]
      : [];
  }
  if (requested !== true) return [];
  for (const feature of ["subgroups", "subgroup-size-control"]) {
    if (!adapterFeatures.has(feature)) {
      throw new GpuFeatureError(
        `subgroups: true was specified, but the adapter lacks '${feature}'`,
      );
    }
  }
  if (!languageFeatures.has("subgroup_id")) {
    throw new GpuFeatureError("subgroups: true requires the WGSL language feature 'subgroup_id'");
  }
  return ["subgroups", "subgroup-size-control"];
};

/**
 * Runs the known answer of a 32-lane reduction, 8 partial sums and a broadcast on all 256 lanes
 * (ADR 0100).
 */
export const assertSubgroup32Executes = async (device: GPUDevice): Promise<void> => {
  const code = `enable subgroups, subgroup_size_control;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 8>;
@compute @workgroup_size(256) @subgroup_size(32)
fn main(@builtin(local_invocation_index) lid: u32,
  @builtin(subgroup_invocation_id) sub: u32, @builtin(subgroup_id) sg: u32,
  @builtin(num_subgroups) count: u32, @builtin(subgroup_size) size: u32) {
  let value = subgroupAdd(f32(lid + 1u));
  if (subgroupElect()) { partial[sg] = value; }
  workgroupBarrier();
  var across = 0.0;
  if (sub < count) { across = partial[sub]; }
  let sum = subgroupAdd(across);
  out[lid] = select(-1.0, sum, size == 32u && count == 8u);
}`;
  const out = device.createBuffer({
    size: 1024,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: 1024,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  try {
    await raceCanaryDeviceLost(
      device,
      withPipelineScope(device, "subgroups canary", () => {
        const pipeline = device.createComputePipeline({
          layout: "auto",
          compute: {
            module: device.createShaderModule({ code }),
            entryPoint: "main",
          },
        });
        const bindings = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: out } }],
        });
        const encoder = device.createCommandEncoder(),
          pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, 1024);
        device.queue.submit([encoder.finish()]);
      }),
      "execution",
      "subgroups",
    );
    await raceCanaryDeviceLost(device, staging.mapAsync(MAP_MODE.READ), "readback", "subgroups");
    const observed = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    if (observed.length !== 256 || observed.some((value) => value !== 32896)) {
      throw new GpuFeatureError(
        "subgroups canary: the 32-lane reduction does not match the known answer",
      );
    }
  } finally {
    // The command was submitted before the readback. Even on failure, no reference to unsubmitted
    // work is left behind.
    staging.destroy();
    out.destroy();
  }
};

/**
 * Measures whether the device maps `local_invocation_index` linearly onto subgroups (lane =
 * index % 32, subgroup = index / 32 at `@subgroup_size(32)`, 256 threads). WGSL defines no such
 * relationship; the online attention's subgroup form (ADR 0132 decision 5) is 16% faster when it
 * can address its tiles by the index, so the fact is measured here and the portable form is used
 * where it does not hold. Runs only after the subgroup features were granted.
 */
export const probeSubgroupLaneMapping = async (device: GPUDevice): Promise<boolean> => {
  const code = `enable subgroups, subgroup_size_control;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) @subgroup_size(32)
fn main(@builtin(local_invocation_index) lid: u32,
  @builtin(subgroup_invocation_id) sub: u32, @builtin(subgroup_id) sg: u32) {
  out[lid] = select(0.0, 1.0, sub == lid % 32u && sg == lid / 32u);
}`;
  const out = device.createBuffer({
    size: 1024,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: 1024,
    usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
  });
  try {
    await raceCanaryDeviceLost(
      device,
      withPipelineScope(device, "subgroup lane-mapping probe", () => {
        const pipeline = device.createComputePipeline({
          layout: "auto",
          compute: {
            module: device.createShaderModule({ code }),
            entryPoint: "main",
          },
        });
        const bindings = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: out } }],
        });
        const encoder = device.createCommandEncoder(),
          pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, 1024);
        device.queue.submit([encoder.finish()]);
      }),
      "execution",
      "subgroups",
    );
    await raceCanaryDeviceLost(device, staging.mapAsync(MAP_MODE.READ), "readback", "subgroups");
    const observed = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return observed.length === 256 && observed.every((value) => value === 1);
  } finally {
    out.destroy();
    staging.destroy();
  }
};

export type DeviceLostHandler = (info: GPUDeviceLostInfo) => void;

export type AcquireGpuOptions = {
  /**
   * The `GPU` instance to request the adapter from. Default: `navigator.gpu`. Node applications
   * pass the instance of a WebGPU implementation here (e.g. Dawn via the npm package `webgpu`:
   * `create([])`) instead of defining `navigator.gpu` themselves; browsers and Deno leave it unset.
   */
  readonly gpu?: GPU;
  /** Passed to requestAdapter as is (powerPreference and the like). */
  readonly adapter?: GPURequestAdapterOptions;
  /** Where unexpected device loss is reported. Not called for a loss caused by `destroy()`. */
  readonly onDeviceLost?: DeviceLostHandler;
  /**
   * Request for GPU-side time measurement (per-op breakdown — ADR 0021). **Three values**; features
   * can be requested only at device creation, so this is the only opt-in point:
   *
   * - `undefined` (default) = **do not request**. No automatic decision — a device with measurement
   *   enabled opens 1 dispatch = 1 pass, so **wall time would change** with the adapter's
   *   capabilities (measurements in the doc of {@link planTimestampFeature}). Diagnostics are paid
   *   for explicitly; that is the default.
   * - `true` = required. {@link GpuFeatureError} on an adapter without it. That it was enabled is
   *   observed through {@link GpuContext.gpuTimingEnabled} and
   *   `Session.diagnostics().lastRunTiming`.
   * - `false` = explicitly not requested (the same behavior as `undefined`; it exists to state "it
   *   is off" in a control run of a measurement).
   */
  readonly gpuTiming?: boolean;
  /**
   * Request for f16 **compute** variants (ADR 0028). The three-value form and the meaning of the
   * default match {@link AcquireGpuOptions.gpuTiming} (for both, `undefined` = do not request):
   *
   * - `undefined` (default) = **do not request**. No automatic decision — f16 compute changes the
   *   numbers, so if the adapter's capabilities decided enabled / disabled, "change the machine and
   *   the output silently changes" would follow.
   * - `true` = required. {@link GpuFeatureError} on an adapter without it. When it could be
   *   enabled, **a tiny f16 kernel with a known answer is run for 1 dispatch and compared** (a
   *   listed feature is no evidence that it "works" — denoland/deno#23125).
   * - `false` = explicitly not requested (the same behavior as `undefined`; it exists only to state
   *   the intent).
   *
   * Whether it could be enabled is observed through {@link GpuContext.shaderF16Enabled}.
   * Passing `"f16"` to `SessionOptions.attentionCompute` / `linearCompute` requires this to be
   * `true` (fails loudly at Session construction).
   */
  readonly shaderF16?: boolean;
  /**
   * Request for 32-lane subgroups (default false, ADR 0100). Requires subgroups and
   * subgroup-size-control, and WGSL's subgroup_id. Whenever the features are requested, a
   * known-answer run is checked at acquisition and a mismatch is rejected.
   *
   * - `true` = required. A shortfall is rejected ({@link GpuFeatureError}).
   * - `"available"` = requested when the adapter has all three, otherwise the device is acquired
   *   without them (no error — ADR 0132 decision 5, amending ADR 0100 decision 1). For code whose
   *   subgroup form is bit-identical to its portable form, such as the online attention kernel
   *   (`attentionScoreStorage: "none"`), which picks its form from the device it gets.
   * - `false` = not requested (the same behavior as `undefined`).
   *
   * Acquiring the capability alone does not change the compute path of the other knobs. A
   * Session's rmsNormReduce / linearGemvReduce subgroup forms are specified separately and still
   * reject a device without the features.
   */
  readonly subgroups?: boolean | "available";
  /**
   * Injection of one static GEMM geometry profile (DECIDED: ADR 0115). The way for a user on an
   * unknown device to apply the table from a sweep run on their own machine (the output of
   * `tools/geometry-sweep`, removed in 0.15).
   *
   * - `undefined` (default) = pick one embedded table from the adapter's (vendor, architecture,
   *   description) (the default profile when none matches). Embedded tables without `match` are
   *   never picked automatically (injection only — look one up by id in
   *   `BUILTIN_GEOMETRY_PROFILES` and pass it here).
   * - a table = **use this table** without looking at the adapter. `match` is not used for matching
   *   (there is a use for applying another machine's table for an A/B).
   * - a function (DECIDED: ADR 0117 decision 6) = the way to look a table up from the information
   *   of the adapter the runtime actually picked (the same value that becomes
   *   `GpuContext.adapterInfo`). Called once per `acquireGpu`, after the adapter is taken and
   *   before the device is created. If it returns a table, that table is injected (going through
   *   the same guard and copy as a table passed directly); `undefined` means the same automatic
   *   selection as no setting. An exception it throws is propagated as is and no device is created.
   *   The runtime does not match the returned table against the adapter (to match, call
   *   `geometryProfileMismatch` of `@karume/runtime/tune` inside the callback).
   *   MUST: it is a synchronous pure function (I/O such as loading a saved table is done before
   *   `acquireGpu`). An adapter may expire within seconds to minutes, and `requestDevice` on an
   *   expired adapter returns a device that is lost from birth instead of throwing, so no window is
   *   created in which we wait while holding the adapter. Returning a Promise (thenable) is
   *   rejected with {@link GpuFeatureError} without awaiting it.
   *
   * It is a property of the device, so it is passed per GPU (it is not in `SessionOptions`). Every
   * Session on the same device and the dp4a canary of i8a8 attention use the same table, and it is
   * never re-chosen during execution (the run-time autotuning ban of the ADR 0022 addendum).
   * Geometry changes only the work assignment and the reduction order over K is unchanged, but
   * output identity is an empirical proposition checked by the sweep's check (output identical to
   * the default) — it is not something the runtime guarantees for an arbitrary table a user brings.
   * The `id` of the chosen table is observed through `Session.diagnostics().geometryProfile`.
   *
   * The passed table is copied and kept (if the caller rewrites it later, the table does not change
   * for the device's lifetime).
   * MUST: a broken table (missing fields, wrong types, an empty id, a malformed `match`, `gemmRows`
   * not ascending / not ending in Infinity, a geometry whose divisibility is broken) fails with
   * {@link GpuFeatureError} **before the device is created** (no device is created from a broken
   * table — the failure comes from user input, so it is a public error type). A table passed
   * directly fails before the adapter is taken, a callback's return value right after the call.
   * Next, for either table, the workgroup shapes (thread count, the components of
   * `@workgroup_size`, shared memory) are checked against the adapter's limits (the planned
   * requiredLimits), and a geometry beyond them also fails with {@link GpuFeatureError} before the
   * device is created (naming the field, the geometry and the limit).
   */
  readonly geometryProfile?:
    | GeometryProfile
    | ((adapterInfo: GPUAdapterInfo) => GeometryProfile | undefined);
  /** Test only ({@link LIMIT_CAPS}). Works only in the direction of **lowering** requiredLimits. */
  readonly [LIMIT_CAPS]?: LimitCaps;
};

/**
 * **A non-public, test-only surface** (not exported from mod.ts — the same style as the executor's
 * `I8A8_DOT`).
 *
 * Creates a device with lowered requiredLimits. The portability check can only be set up this way —
 * local adapters report values far above the WebGPU core defaults (`maxStorageBufferBindingSize` =
 * 128MiB), so "do allocation and binding pass on a machine with core defaults" can only be checked
 * by **actually running on a lowered device** (the same discipline as "a listed feature is no
 * evidence that it works").
 */
export const LIMIT_CAPS: unique symbol = Symbol("karume.limitCaps");

/**
 * Takes the adapter from `host` (`navigator.gpu` when omitted) (shared by {@link acquireGpu} and
 * {@link readAdapterLimits}). An environment without one gives {@link GpuUnavailableError} — kept
 * in this one place so the two entry points share the same diagnostic wording.
 */
const requestAdapterOrThrow = async (
  options: GPURequestAdapterOptions | undefined,
  host: GPU | undefined,
): Promise<{ readonly gpu: GPU; readonly adapter: GPUAdapter }> => {
  const gpu: GPU | undefined = host ?? navigator.gpu;
  if (gpu === undefined) {
    throw new GpuUnavailableError(
      "navigator.gpu does not exist (an environment without WebGPU; on Node, pass an implementation in AcquireGpuOptions.gpu)",
    );
  }
  const adapter = await gpu.requestAdapter(options);
  if (adapter === null) {
    throw new GpuUnavailableError("Cannot obtain a GPUAdapter (no supported GPU / driver)");
  }
  return { gpu, adapter };
};

/**
 * Runs adapter acquisition → raising limits → device acquisition → validating the result.
 * Every failure along the way is an exception (a device with silently reduced capabilities is never
 * returned).
 */
export const acquireGpu = async (options: AcquireGpuOptions = {}): Promise<GpuContext> => {
  const source = options.geometryProfile;
  // The guard for a table passed directly runs before touching either the adapter or the device (a
  // check of input that does not depend on the GPU). It is copied so that the table does not change
  // for the device's lifetime even if the caller rewrites it later (Infinity survives
  // structuredClone).
  const givenProfile =
    source === undefined || typeof source === "function"
      ? undefined
      : cloneCheckedGeometryProfile(source, DIRECT_PROFILE_LABEL);
  const { gpu, adapter } = await requestAdapterOrThrow(options.adapter, options.gpu);
  // Read once for both the value passed to the callback and GpuContext.adapterInfo (reading twice
  // would make it possible for the adapter the callback looked up and the device's adapter to
  // disagree)
  const adapterInfo = readAdapterInfo(adapter);
  // MUST: call the callback once, after taking the adapter and before creating the device (ADR 0117
  // decision 6). An exception it throws is propagated as is, and no device is created along with a
  // broken table
  const geometryProfile =
    typeof source === "function" ? profileFromCallback(source(adapterInfo)) : givenProfile;
  const limits = planRequiredLimits(adapter.limits, options[LIMIT_CAPS]);
  // Guard order is structure (the copy guard above) → limits. Limits need the adapter, so a table
  // passed directly is also checked here. The values checked are the planned limits (= what the
  // device will have — including the reducing caps)
  if (geometryProfile !== undefined) {
    try {
      assertGeometryProfileWithinLimits(geometryProfile, limits);
    } catch (error) {
      throw publicProfileError(
        typeof source === "function" ? CALLBACK_PROFILE_LABEL : DIRECT_PROFILE_LABEL,
        error,
      );
    }
  }
  // Conditional features are decided only here (a shortfall is an exception — capabilities are
  // never silently dropped). ADR 0021 / 0028.
  const timestampQuery = planTimestampFeature(adapter.features, options.gpuTiming);
  const shaderF16 = planShaderF16Feature(adapter.features, options.shaderF16);
  const languageFeatures = readWgslLanguageFeatures(gpu);
  const subgroupFeatures = planSubgroup32Features(
    adapter.features,
    languageFeatures,
    options.subgroups,
  );
  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...REQUIRED_FEATURES,
      ...(timestampQuery ? [TIMESTAMP_QUERY_FEATURE] : []),
      ...(shaderF16 ? [SHADER_F16_FEATURE] : []),
      // A WebGPU feature not in Deno's type definitions. Limited to the boundary where the
      // adapter's advertisement was checked above.
      ...(subgroupFeatures as readonly GPUFeatureName[]),
    ],
    requiredLimits: limits,
  });
  let subgroupLanesLinear = false;
  try {
    assertLimitsGranted(limits, device.limits);
    // MUST: confirm by running, not by the list (the silent all-0 of denoland/deno#23125).
    if (shaderF16) await assertShaderF16Executes(device);
    if (subgroupFeatures.length > 0) await assertSubgroup32Executes(device);
    // The online attention's subgroup form assumes the linear lane mapping (ADR 0132 decision 5):
    // measured, never assumed — a device whose mapping differs runs the portable form.
    subgroupLanesLinear = subgroupFeatures.length > 0 && (await probeSubgroupLaneMapping(device));
  } catch (cause) {
    device.destroy();
    throw cause;
  }
  return new GpuContext(
    device,
    adapterInfo,
    limits,
    languageFeatures,
    options.onDeviceLost,
    geometryProfile,
    subgroupLanesLinear,
  );
};

/**
 * The prefix of the injected-table guard's message (a table passed directly / a callback's return
 * value).
 */
const DIRECT_PROFILE_LABEL = "geometryProfile";
const CALLBACK_PROFILE_LABEL = "geometryProfile (callback return value)";

/**
 * Turns a failure of the injected-table guard into the public {@link GpuFeatureError} instead of
 * the internal {@link CodegenError} (failures from user input can be handled with a public type —
 * the same style as the `subgroups` check). Any other exception is returned as is.
 */
const publicProfileError = (label: string, error: unknown): unknown =>
  error instanceof CodegenError ? new GpuFeatureError(`${label}: ${error.message}`) : error;

/**
 * Runs the structural guard for an injected table ({@link assertGeometryProfile} — narrows unknown
 * to a table) and returns a copy of the table that passed.
 */
const cloneCheckedGeometryProfile = (profile: unknown, label: string): GeometryProfile => {
  try {
    assertGeometryProfile(profile);
  } catch (error) {
    throw publicProfileError(label, error);
  }
  return structuredClone(profile);
};

/**
 * A value that has `then` as a function (a Promise, or a home-made thenable that claims to be one).
 */
const isThenable = (value: unknown): boolean =>
  typeof value === "object" &&
  value !== null &&
  "then" in value &&
  typeof value.then === "function";

/**
 * A no-op so that no Promise is created whose rejection nobody catches ({@link
 * profileFromCallback}).
 */
const ignoreRejection = (): void => {};

/**
 * Attaches a no-op rejection handler only to a real Promise (from any realm). Never calls the
 * `then` of a home-made thenable.
 *
 * `instanceof Promise` misses Promises from another realm (an iframe, `node:vm`), and
 * `Symbol.toStringTag` can be faked. This realm's built-in `Promise.prototype.then` rejects a value
 * that is not a real Promise with TypeError **before reading either then or constructor**, through
 * a brand check on the argument's internal slots (IsPromise in the specification), so it serves as
 * a realm-independent test and the attachment in one go. Only that TypeError is caught (not real =
 * nothing to attach to) — anything else propagates.
 */
const ignoreRejectionIfNativePromise = (value: unknown): void => {
  try {
    // Fire-and-forget: the derived promise only exists to mark the rejection as handled.
    // oxlint-disable-next-line typescript/unbound-method -- Reflect.apply supplies `value` as the receiver; that receiver is the brand check itself
    void Reflect.apply(Promise.prototype.then, value, [undefined, ignoreRejection]);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
  }
};

/**
 * Passes a callback's return value through the same guard and copy as a table passed directly
 * (`undefined` = automatic selection).
 *
 * MUST: reject a Promise (thenable) without awaiting it. Waiting would open an I/O window while
 * holding the adapter, and in the meantime a device lost from birth could be created from an
 * expired adapter — a silent failure (ADR 0117 decision 6).
 * The guard for values that arrive outside the type (a synchronous function) — JS callers, async
 * functions.
 */
const profileFromCallback = (
  returned: GeometryProfile | undefined,
): GeometryProfile | undefined => {
  if (isThenable(returned)) {
    // The reject of an async function becomes an unhandled rejection — the reject of a Promise
    // nobody waits for after we have refused it (on Deno the process dies). Attach a no-op
    // rejection handler only to real Promises (including those from another realm) — no waiting,
    // the result is not used. The then of a home-made thenable is never called (no arbitrary code
    // runs)
    ignoreRejectionIfNativePromise(returned);
    throw new GpuFeatureError(
      "geometryProfile: the callback returned a Promise (it must be a synchronous function — load the saved table " +
        "before acquireGpu, and let the callback only pick a table from the loaded values)",
    );
  }
  return returned === undefined
    ? undefined
    : cloneCheckedGeometryProfile(returned, CALLBACK_PROFILE_LABEL);
};

/**
 * Reads only the adapter's limits (no device is created).
 *
 * The entry point for checking the `requiredLimits` a distribution declares (the quant field of the
 * manifest) **before downloading the weights** (ADR 0089 decision 5). The return value is the same
 * formula as the limits {@link acquireGpu} plans with the same arguments ({@link
 * planRequiredLimits}, without reductions), so callers can compare with the same yardstick as
 * `GpuContext.limits` after acquisition.
 *
 * MUST: discard the adapter after reading (do not carry it around). A WebGPU adapter "may expire at
 * any time" (specification §4.2 — it may expire within seconds to minutes even without a change in
 * system state), and `requestDevice` after expiry is a silent failure that returns a device lost
 * from birth instead of throwing. The device is still created by {@link acquireGpu} from an adapter
 * it has just taken again.
 *
 * NOTE: the specification does not guarantee that calling `requestAdapter` twice picks the same
 * physical adapter. The limits read here are material for a preliminary check; the final check is
 * the real buffer check at Session construction (`assertWeightsWithinLimits`).
 */
export const readAdapterLimits = async (
  options: Pick<AcquireGpuOptions, "adapter" | "gpu"> = {},
): Promise<RequiredLimits> => {
  const { adapter } = await requestAdapterOrThrow(options.adapter, options.gpu);
  return planRequiredLimits(adapter.limits);
};
