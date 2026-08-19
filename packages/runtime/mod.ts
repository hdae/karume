/**
 * Karume の公開 API。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — GPU を取る / モデルを開く / 能力を照会する /
 * セッションを作る / 実行する / 解放する / 診断を得る / 失敗を型で捌く。
 *
 * エラー型は「利用者の入力・環境に起因する失敗」だけを公開する。ランタイム内部の不変条件
 * 破れ（アリーナの参照計数・codegen 決定性など）は Karume 側のバグであって利用者が
 * 分岐すべきものではないため、`Error`（`name` は保持）のまま届く。
 */

export { ContainerError, openModel } from "./src/format/container.ts";
export type { KarumeModel } from "./src/format/container.ts";
export { DimError } from "./src/format/dims.ts";
export { IrError } from "./src/format/ir.ts";
/**
 * safetensors 厳格リーダ（被覆・整列・dtype 検査込み）。IR コンテナでない付帯資産
 * （例: Anima の rope 素表 — ADR 0038 §2 の extras）を models 側が同じ門で読むための面で、
 * 汎用ローダの提供が目的ではない（DECIDED: 二重実装の解消 — ADR 0008 追記 2026-08-05）。
 */
export { parseSafetensors, SafetensorsError, tensorBytes } from "./src/format/safetensors.ts";
export type { SafetensorsDtype, SafetensorsFile, TensorView } from "./src/format/safetensors.ts";

export {
  acquireGpu,
  BatchScopeError,
  GpuDeviceLostError,
  GpuFeatureError,
  GpuLimitError,
  GpuOutOfMemoryError,
  GpuUnavailableError,
  GpuValidationError,
  ResidentTensorError,
} from "./src/gpu/device.ts";
/**
 * GpuContext の構築は {@link acquireGpu} だけを入口にするため、型としてのみ公開する
 * （Session と同じ形）。値として公開すると `new GpuContext(...)` で planRequiredLimits /
 * assertLimitsGranted を迂回した device を渡せてしまい、limits 要求漏れの検出網が抜ける。
 *
 * `ResidentTensor`（GpuContext 所有の第 4 の寿命クラス）と `BatchScope`（フェンス 1 本で閉じる
 * enqueue 区間）も同じ理由で型としてのみ公開する — 入口は `GpuContext.createResident` /
 * `GpuContext.beginBatch` だけで、直接構築すると errorScope の門（確保失敗の検出・区間ロックの
 * 保持）を迂回できてしまう。
 */
export type {
  AcquireGpuOptions,
  BatchScope,
  DeviceLostHandler,
  GpuContext,
  ResidentData,
  ResidentTensor,
} from "./src/gpu/device.ts";

export type { ArenaStats } from "./src/gpu/arena.ts";
export { DEFAULT_SUBMIT_POLICY, SubmitPolicyError } from "./src/gpu/submit.ts";
/**
 * GPU 時間内訳（ADR 0021）は型だけ公開する。集計は SubmitScheduler の内部で、利用者の面は
 * `Session.diagnostics().lastRunTiming`（計測が無効なら undefined）1 つに閉じる。
 */
export type {
  GpuTimingEntry,
  GpuTimingStats,
  SubmitPolicy,
  SubmitStats,
} from "./src/gpu/submit.ts";

export { capabilities, OpContractError } from "./src/ops.ts";
export type { RuntimeCapabilities } from "./src/ops.ts";

export { DispatchLimitError } from "./src/codegen/errors.ts";

export { createSession, createSessionFromShards } from "./src/runtime/executor.ts";
/**
 * メモリ必要量 estimator（ADR 0070 決定 5）。GPU 非依存の純関数で「必要側」のカテゴリ別
 * バイト数だけを出す — 空き側との比較・可否判定はしない（最終門は out-of-memory errorScope）。
 */
export { estimateSessionMemory } from "./src/runtime/estimate.ts";
export type { EstimateOptions, MemoryEstimate } from "./src/runtime/estimate.ts";
/**
 * Session の構築は {@link createSession} だけを入口にするため、型としてのみ公開する。
 *
 * `Tensor` は意味論 dtype の判別ユニオン（ADR 0009 による ADR 0008 の部分改訂）:
 * `{ dtype: "f32", data: Float32Array }` / `{ dtype: "i32", data: Int32Array }` /
 * `{ dtype: "bool", data: Uint32Array }`（bool は u32 の 0 / 1）。入出力で対称。
 *
 * `GenerationContext`（1 生成ぶんの可変 state の所有者 — ADR 0066）も同じ理由で型のみ:
 * 入口は `Session.createGenerationContext` だけで、直接構築すると確保の errorScope と容量
 * ゲートを迂回できてしまう（`ResidentTensor` / `BatchScope` と同じ流儀）。
 */
export type {
  ComputePrecision,
  EnqueueOptions,
  FusionCounts,
  GenerationContext,
  GenerationContextSpec,
  ParamsCacheStats,
  PlanBackingStats,
  PreparedPlanStats,
  RunInput,
  RunInputs,
  RunOutputs,
  ScoreStorage,
  Session,
  SessionDiagnostics,
  SessionOptions,
  StateBackingStats,
  StorageDiagnostics,
  Tensor,
} from "./src/runtime/executor.ts";
export { ExecutionError } from "./src/runtime/plan.ts";
export type { SymbolBindings } from "./src/runtime/plan.ts";
