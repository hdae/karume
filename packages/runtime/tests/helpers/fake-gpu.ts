// フェイク device を**実物の** GpuContext に包むためのヘルパ。
//
// MUST: GpuContext 自体はフェイクにしない。検証したいのは errorScope 区間ロック・消失購読と
// いった GpuContext の本番実装そのもので、そこを差し替えるとテストがフェイクを検証するだけに
// なる。差し替えるのは GPU 実体（device）だけに留める。

import {
  type DeviceLostHandler,
  GpuContext,
  readAdapterInfo,
  type RequiredLimits,
} from "../../src/gpu/device.ts";

/** ロックと消失購読は limits を見ないため、要求キーを 0 で埋めた最小値でよい。 */
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

/** GpuContext が構築時に触る面（features / lost）だけを持つ最小のフェイク device。 */
export type FakeDeviceParts = {
  /** 未指定なら「消失しない device」（永久に未解決の promise）。 */
  readonly lost?: Promise<GPUDeviceLostInfo>;
  readonly features?: Iterable<string>;
};

/** DOM 型全体は再現しないため cast で渡す（テスト専用の境界）。 */
export const fakeDevice = (parts: FakeDeviceParts = {}): GPUDevice =>
  ({
    lost: parts.lost ?? new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set(parts.features ?? []),
  }) as unknown as GPUDevice;

export const fakeGpuContext = (
  device: GPUDevice,
  onDeviceLost?: DeviceLostHandler,
): GpuContext => new GpuContext(device, readAdapterInfo({}), ZERO_LIMITS, new Set(), onDeviceLost);

/**
 * 消失を後から起こせるフェイク device と、その引き金。
 * `reason` は GpuContext の分岐（意図的な破棄かどうか）に使われないため固定値でよい。
 */
export const losableGpuContext = (): {
  readonly gpu: GpuContext;
  readonly lose: () => void;
} => {
  let resolve: (info: GPUDeviceLostInfo) => void = () => {};
  const lost = new Promise<GPUDeviceLostInfo>((settle) => {
    resolve = settle;
  });
  return {
    gpu: fakeGpuContext(fakeDevice({ lost })),
    lose: (): void => resolve({ reason: "destroyed", message: "テストによる消失" }),
  };
};
