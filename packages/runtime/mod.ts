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

/**
 * Karume 専用コンテナ（`krm` / `krg` — ADR 0108・docs/container-v1.md）の読み口。`openContainer` は
 * 外側の期待 hash で 2 文書を検証してから parse し、グラフと束縛表を合流する（重みの block は
 * 要るときに取る）。`codecLayout` は codec 台帳の登録名 → 展開経路（`ternary` は `int2-off` と同じ
 * i2 経路）で、models が格納の性質で分岐するときの唯一の読み口。
 */
export { openContainer } from "./src/format/container/open.ts";
export type {
  AssetReader,
  BlockSource,
  ContainerInput,
  DescriptorExpectation,
  OpenedContainer,
} from "./src/format/container/open.ts";
/**
 * **メモリ内容器** — 手元のバイト列（合成した重み・別の器から読み出した実体）を容器を書かずに
 * Session 構築へ渡す供給面。`openContainer` と同じ 2 つの読み口（{@link BoundContainer}）を返すので、
 * `prepareContainer` / `createSessionFromContainer` はどちらの供給元でも同じ 1 本を通る。
 *
 * `MemoryTensor` は丸ごと 1 本（`bytes`）と行分割（`pieces` — 読み口の列）の排他で、後者は
 * バイト列を 1 つも抱えない（part ごとのフェンスで参照が尽きる = 器 1 本ぶんの RAM）。
 */
export { openMemoryContainer } from "./src/format/container/memory.ts";
export type {
  MemoryContainerInput,
  MemoryEncoding,
  MemoryPiece,
  MemoryTensor,
} from "./src/format/container/memory.ts";
/** 供給元 2 種（`krm` / メモリ内容器）が満たす共通の面。置き場は合流層（供給元の実装ではない）。 */
export type { BoundContainer } from "./src/format/container/bind.ts";
export { ContainerFormatError } from "./src/format/container/header.ts";
export { codecLayout } from "./src/format/container/codecs.ts";
export type { CodecLayout, CodecName } from "./src/format/container/codecs.ts";
export { DimError } from "./src/format/dims.ts";
export { IrError } from "./src/format/ir.ts";
/**
 * **ホストで組む IR v2 宣言の入口**（docs/ir-v2.md）。exporter を通さずその場で組んだ小さな
 * グラフ（CFG 合成・Euler 更新・最大値選択・PLE gather）を {@link openMemoryContainer} へ渡す
 * 呼び手が、**グラフ単体で決まる規則**をここで通す — SSA・前方参照・トポロジカル順・
 * `requires.ops` と実使用 op の一致・未知キー・記号名の正準表記・孤立宣言。
 *
 * この門は `krm` を開く経路（`openContainer`）が容器の宣言に対して通すものと同じ 1 本である。
 * 公開していないと、メモリ内容器の呼び手だけが検査を迂回した宣言で Session を組めてしまい、
 * 壊れたグラフは合流（宣言 × 供給）か実行まで落ちてこない。
 *
 * `initializers[name]` は `{}`（実体を持つ）か `{ "shared": true }`（貸し手 Session の重みを
 * 借りる）の 2 形だけ（`"shared": false` は書けない — 欄の不存在と同じ宣言なので正準形が
 * 2 通りになる）。`states` とノードの `states` は省略できる（空表が既定）。
 *
 * NOTE: 受けるのは **JSON.parse 済みの素の値**で、`1e999` 由来の非有限数と入れ子の深さは
 * ここでは見ない — `krm` 経路ではその 2 つを記述文書の読み手（`openContainer`）が持ち、
 * ホストで組む呼び手ではリテラルを書く側が持つ。外から来た JSON 文字列を載せるなら、
 * この 2 つを自分で検査してから渡す MUST（素の `JSON.parse` の戻りをそのまま渡すと
 * `krm` 経路にある検査だけが抜ける）。
 */
export { parseIrDeclarationValue } from "./src/format/ir.ts";
/** {@link parseIrDeclarationValue} が返す形（= `MemoryContainerInput["graphs"]` の値）。 */
export type { IrDeclaration } from "./src/format/ir.ts";
/**
 * safetensors 厳格リーダ（被覆・整列・dtype 検査込み）。Karume の容器でない付帯資産
 * （例: Anima の rope 素表 — `transformer` 容器の資産 `rope_base`〈役割 `rope-base` — ADR 0109
 * 決定 4〉の payload）を models 側が同じ門で読むための面で、
 * 汎用ローダの提供が目的ではない（DECIDED: 二重実装の解消 — ADR 0008 追記 2026-08-05）。
 */
export { parseSafetensors, SafetensorsError } from "./src/format/safetensors.ts";
export type { SafetensorsDtype, SafetensorsFile, TensorView } from "./src/format/safetensors.ts";

export {
  acquireGpu,
  BatchScopeError,
  GpuDeviceLostError,
  GpuFeatureError,
  GpuLimitError,
  GpuUnavailableError,
  readAdapterLimits,
  ResidentTensorError,
} from "./src/gpu/device.ts";
/** errorScope の規律から出る型付きエラー（置き場は `src/gpu/error-scope.ts` — 公開面は不変）。 */
export {
  GpuInternalError,
  GpuOutOfMemoryError,
  GpuValidationError,
} from "./src/gpu/error-scope.ts";
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
  RequiredLimits,
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
  ChunkBudgetStats,
  GpuTimingEntry,
  GpuTimingStats,
  SubmitPolicy,
  SubmitStats,
} from "./src/gpu/submit.ts";

export { capabilities, OpContractError } from "./src/ops.ts";
export type { RuntimeCapabilities } from "./src/ops.ts";
/**
 * capability 不足（実行できない op / 意味論 dtype / attrs / 格納）。`prepareContainer` が
 * **重みの block を 1 つも取る前に**全件列挙して投げる型（ADR 0070 決定 5 の 2 段境界）。
 */
export { RuntimeSupportError } from "./src/ops/support.ts";

export { DispatchLimitError } from "./src/codegen/errors.ts";

export { createSessionFromContainer, prepareContainer } from "./src/runtime/executor.ts";
export { DEFAULT_PLAN_BACKING_BUDGET_BYTES } from "./src/runtime/session-types.ts";
/**
 * 重み取得前の admission の入口（ADR 0070 決定 5 / graph-first）。開いた容器のグラフ宣言だけで
 * 「実行できない」を先に落とし、必要メモリを見積り、そのまま Session にする 2 段境界:
 * `prepareContainer(opened, graphName) → estimate() → createContainerSession(gpu)`。
 *
 * `PreparedModel` は**型としてのみ**公開する — 入口は {@link prepareContainer} だけで、直接構築
 * すると capability 門と IR 契約検査を迂回した「実行できないモデルの Session」が作れてしまう
 * （`Session` / `GpuContext` と同じ流儀）。
 */
export type { PreparedModel } from "./src/runtime/executor.ts";
/**
 * メモリ必要量 estimator（ADR 0070 決定 5）。GPU 非依存の純関数で「必要側」のカテゴリ別
 * バイト数だけを出す — 空き側との比較・可否判定はしない（最終門は out-of-memory errorScope）。
 */
export { estimateGraphMemory } from "./src/runtime/estimate.ts";
export type {
  AdmissionReport,
  AdmissionScenario,
  AdmissionScenarioName,
  AdmissionScenarioSpec,
  EstimateOptions,
} from "./src/runtime/estimate.ts";
/**
 * 重み常駐計画 — グラフだけで決まる純関数で、{@link estimateGraphMemory} の第 2 引数。
 *
 * ロードを終えた呼び手（models のパイプライン）は `PreparedModel` を握らず `IrGraph` だけ
 * 残す規律なので、`PreparedModel.estimate` の口が使えない。そこから同じ見積りへ戻る唯一の
 * 経路がこの 2 本（`estimateGraphMemory(graph, planWeightResidency(graph), options)`）で、
 * グラフが同じなら計画も同じ（純関数）。
 *
 * グラフは `PreparedModel.graph`（構築と同じ門を通った検証済みのグラフ）から取る — この呼び方
 * では常駐計画が見積りの門より先に走るので、未検証のグラフでは失敗の文言がどのノードかを名乗らない。
 */
export { planWeightResidency } from "./src/runtime/weight-residency.ts";
export type { WeightResidency } from "./src/runtime/weight-residency.ts";
/**
 * Session の構築は {@link prepareContainer} の `createContainerSession` と、その合成である
 * {@link createSessionFromContainer} だけを入口にするため、型としてのみ公開する。
 *
 * `Tensor` は意味論 dtype の判別ユニオン（ADR 0009 による ADR 0008 の部分改訂）:
 * `{ dtype: "f32", data: Float32Array }` / `{ dtype: "i32", data: Int32Array }` /
 * `{ dtype: "bool", data: Uint32Array }`（bool は u32 の 0 / 1）。入出力で対称。
 *
 * 入力の寿命は面の一部: `run` / `enqueue` は `inputs` Record の member 構成・各入力の shape・
 * `bindings` を**発行の同期区間で写し取る**が、`Tensor.data` は写さず**借りる** —
 * MUST NOT: 戻り Promise が settle するまで `data` を書き換える（沈黙誤値になる）。詳細は
 * `Session.run` の「入力の寿命」節。
 *
 * `GenerationContext`（1 生成ぶんの可変 state の所有者 — ADR 0066）も同じ理由で型のみ:
 * 入口は `Session.createGenerationContext` だけで、直接構築すると確保の errorScope と容量
 * ゲートを迂回できてしまう（`ResidentTensor` / `BatchScope` と同じ流儀）。
 *
 * `SharedWeight`（貸し出された重みへの不透明な参照 — ADR 0096 段 2 §2.2）も同じ理由で型のみ:
 * 入口は `Session.exportWeight` だけで、直接構築すると席の突合と借用計数を迂回できる。
 */
export type {
  ComputePrecision,
  EnqueueOptions,
  EnqueueRead,
  FusionCounts,
  GenerationContext,
  GenerationContextSpec,
  GenerationRun,
  LinearGemvReduce,
  ParamsCacheStats,
  PlanBackingStats,
  PreparedPlanStats,
  RmsNormReduce,
  RunInput,
  RunInputs,
  RunOutputs,
  ScoreStorage,
  Session,
  SessionBuildStats,
  SessionDiagnostics,
  SessionOptions,
  SharedWeight,
  StateAttentionReduce,
  StateBackingStats,
  StorageDiagnostics,
  Tensor,
} from "./src/runtime/executor.ts";
/**
 * `GenerationContextSpec.chunkBuckets` の値域・順序検査（ADR 0066 追記〈バケット〉）。
 *
 * GPU も Session も要らない純関数を**あえて**公開面に出しているのは、prefill 形の物理行数を
 * 選ぶ側（models のパイプライン）が同じ規則を独立に持たないため — 規則を写すと、context が
 * 許す集合と呼び出し側が選ぶ集合が別々に育ち、食い違いは `Session.run` の run 前検査という
 * 真因から遠い場所で出る。受理集合の正本は 1 本だけにする。
 */
export { assertChunkBuckets } from "./src/runtime/generation-context.ts";
export { ExecutionError } from "./src/runtime/plan.ts";
export type { SymbolBindings } from "./src/runtime/plan.ts";
