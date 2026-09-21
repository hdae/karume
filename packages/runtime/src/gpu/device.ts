/**
 * WebGPU デバイスの取得と能力の正規化。
 *
 * この層の責務は「実装差のある面（features / limits / WGSL 言語機能）を 1 箇所で正規化し、
 * 沈黙故障を全て loud な例外に変換すること」。
 *
 * MUST: device をモジュールスコープに捕獲しない。VRAM を返すのは `device.destroy()` のみで
 * （`buffer.destroy()` は 1 バイトも返さない）、解放後の続行は `acquireGpu()` からの再構築に
 * なる。device を握る層（PipelineCache / SubmitScheduler / RunArena）は全て GpuContext と
 * 同じ寿命で作り直せる構造でなければならない。
 *
 * **この層の入口**で、実体は 2 つに分かれている: {@link "./acquire.ts"}（device を取って
 * GpuContext を組み立てるまで — アダプタ取得・limits / features の計画と検査・カナリアの実走）
 * と {@link "./context.ts"}（取得済み device の器 — GpuContext / ResidentTensor / BatchScope
 * とランタイム内部面）。ここは両者の公開名を再 export するだけで、実装を持たない。
 *
 * MUST: 実体（値）の依存は **acquire → context の一方向**に保つ。GpuContext を構築するのは
 * `acquireGpu` だけなので逆向きの実体 import は要らず、作れば device 取得・カナリアと器の
 * 寿命管理が循環する（context から acquire への参照は `import type` だけ — 消去されるので
 * 実行時の import グラフは一方向のまま）。
 * MUST: 利用者・runtime の他モジュール・テストは**この綴り（`device.ts`）で import する**。
 * 層の入口を 1 つに保つための facade で、2 ファイルへの分割は実装の都合にすぎない。
 */

export {
  acquireGpu,
  assertLimitsGranted,
  assertShaderF16Executes,
  assertSubgroup32Executes,
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
export type { BatchMember, BatchReadSource, ResidentData } from "./context.ts";
