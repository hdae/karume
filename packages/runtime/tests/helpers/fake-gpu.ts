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

/** 引き金に何も渡さなかったときの消失情報（`reason` は GpuContext の分岐に使われない）。 */
const DEFAULT_LOSS: GPUDeviceLostInfo = { reason: "destroyed", message: "テストによる消失" };

/**
 * 消失を後から起こせるフェイク device と、その引き金。
 *
 * `lose` に情報を渡せるのは、バックエンドが入れる真因（`reason` / `message`）が例外の文言まで
 * 運ばれることを固定するため。
 *
 * `onDeviceLost` を渡せるのは、公開通知が**挿入順の先頭**に来る形（コンストラクタでの登録）
 * を再現するため。内部購読より先に呼ばれることが、例外隔離を要求する条件そのものになる。
 */
export const losableGpuContext = (onDeviceLost?: DeviceLostHandler): {
  readonly gpu: GpuContext;
  readonly lose: (info?: GPUDeviceLostInfo) => void;
} => {
  let resolve: (info: GPUDeviceLostInfo) => void = () => {};
  const lost = new Promise<GPUDeviceLostInfo>((settle) => {
    resolve = settle;
  });
  return {
    gpu: fakeGpuContext(fakeDevice({ lost }), onDeviceLost),
    lose: (info: GPUDeviceLostInfo = DEFAULT_LOSS): void => resolve(info),
  };
};
