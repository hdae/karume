/**
 * 最小のグラフ実行器。
 *
 * 構造: 「計画（純関数・plan.ts）→ **導出**（レシピ構築 — src/runtime/recipe.ts）→ **実行**
 * （レシピ列を汎用ループでエンコード）→ submit → グラフ出力だけ readback」。
 * 出力 shape は 1 dispatch も出す前に全て確定し（静的形状 — ADR 0004）、想定外は全て
 * fail loudly にする（未対応 op / 契約外 dtype / broadcast 不可 / 未束縛シンボル / 宣言不一致）。
 *
 * MUST: 導出相は **run 寿命の状態に触れない**（{@link RunArena} の確保も dispatch の発行も
 * しない）。これが成り立っているので、導出相の成果物は解決済み bindings をキーに Session へ
 * 常駐させられる（{@link PreparedPlan}）— 同一 bindings の 2 run 目以降は計画・融合判定・
 * レシピ導出を丸ごと飛ばし、レシピ列をそのまま実行相へ渡す。
 *
 * MUST: 初期化は明示 async ステージ（{@link createSession}）。コンストラクタ内の同期
 * アップロードループは、重み取得とアップロードのパイプライン化を構造的に不可能にする。
 * MUST: バッファ破棄の前に未 submit のエンコードを必ず片付ける — 成功経路は flush、失敗経路は
 * discard（discard-or-flush before destroy）。破棄済みバッファを参照するエンコードを submit
 * するとコマンドバッファ丸ごと失敗し、同じスケジューラに相乗りしている無関係な dispatch まで
 * 実行されないまま誤った値が静かに残る。
 */

import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../codegen/elementwise.ts";
import {
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceKey,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../codegen/reduce.ts";
import {
  catOutOffset,
  catOutStrides,
  expandSrcStrides,
  permuteSrcStrides,
  prefixSliceSrcStrides,
  sliceSrcOffset,
  sliceSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  STRIDED_WRITE_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
  stridedWriteKey,
  stridedWriteParams,
  stridedWriteWgsl,
} from "../codegen/strided.ts";
import {
  CONV1D_SCALE_BINDING,
  CONV1D_WORKGROUP_SIZE,
  conv1dKey,
  conv1dParams,
  conv1dWgsl,
} from "../kernels/conv1d.ts";
import {
  CONV2D_SCALE_BINDING,
  CONV2D_WORKGROUP_SIZE,
  type Conv2dDims,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dKey,
  conv2dParams,
  conv2dUsesVec4,
  conv2dWgsl,
} from "../kernels/conv2d.ts";
import {
  CONV_TRANSPOSE1D_SCALE_BINDING,
  CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  convTranspose1dKey,
  convTranspose1dParams,
  convTranspose1dWgsl,
} from "../kernels/conv-transpose1d.ts";
import { CUMSUM_KEY, CUMSUM_WGSL, CUMSUM_WORKGROUP_SIZE, cumsumParams } from "../kernels/cumsum.ts";
import { FLIP_KEY, FLIP_WGSL, FLIP_WORKGROUP_SIZE, flipParams } from "../kernels/flip.ts";
import { PAD_KEY, PAD_WGSL, PAD_WORKGROUP_SIZE, padParams } from "../kernels/pad.ts";
import {
  EMBEDDING_SCALE_BINDING,
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../kernels/embedding.ts";
import { LAYER_NORM_KEY, LAYER_NORM_WGSL, layerNormParams } from "../kernels/layer-norm.ts";
import { RMS_NORM_KEY, RMS_NORM_WGSL, rmsNormParams } from "../kernels/rms-norm.ts";
import { LINEAR_SCALE_BINDING, linearKey, linearParams, linearWgsl } from "../kernels/linear.ts";
import {
  dp4aAvailable,
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../kernels/linear-i8a8.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../kernels/quantize-rows.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import {
  MASKED_FILL_KEY,
  MASKED_FILL_WGSL,
  MASKED_FILL_WORKGROUP_SIZE,
  maskedFillParams,
} from "../kernels/masked-fill.ts";
import { SOFTMAX_KEY, SOFTMAX_WGSL, softmaxParams } from "../kernels/softmax.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsRegCache,
  attentionStatsWgsl,
} from "../kernels/attention.ts";
import { defaultI8a8Geometry, i8a8TileM, i8a8TileN } from "../kernels/i8a8-geometry.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../kernels/attention-i8a8.ts";
import {
  attentionScoreUsesF16,
  type ScoreStorage,
  scoreStorageBytes,
} from "../kernels/score-storage.ts";
/** S の格納形（{@link SessionOptions.attentionScoreStorage} — 公開面で名前を持てるように再輸出）。 */
export type { ScoreStorage } from "../kernels/score-storage.ts";
import { assertRuntimeSupport, type KarumeModel } from "../format/container.ts";
import { alignF16Payload, decodeF16 } from "../format/f16.ts";
import { alignI8Payload, decodeI8 } from "../format/i8.ts";
import type { IrDtype } from "../format/ir.ts";
import { type SafetensorsFile, tensorBytes } from "../format/safetensors.ts";
import { type ArenaStats, RunArena, STORAGE_USAGE } from "../gpu/arena.ts";
import {
  discardFailureScopes,
  type GpuContext,
  popFailureScopes,
  pushFailureScopes,
} from "../gpu/device.ts";
import { PipelineCache } from "../gpu/pipeline-cache.ts";
import {
  type GpuTimingStats,
  type SubmitPolicy,
  SubmitScheduler,
  type SubmitStats,
} from "../gpu/submit.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../kernels/bmm.ts";
import {
  ATTENTION_QK_MASK_BINDING,
  type GemmCompute,
  gemmMTileGeometry,
  gemmUsesVec4,
} from "../kernels/gemm.ts";
import {
  defaultGemmGeometry,
  gemmGeometryForRows,
  gemmTileM,
  gemmTileN,
} from "../kernels/gemm-geometry.ts";
import { GATHER_KEY, GATHER_WGSL, GATHER_WORKGROUP_SIZE, gatherParams } from "../kernels/gather.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../kernels/matmul.ts";
import {
  attentionScale,
  type BinaryOpName,
  CAST_OP,
  catDim,
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  flipDim,
  layerNormAttrs,
  maskedFillValue,
  numel,
  padAttrs,
  permuteDims,
  reduceDim,
  type ReduceOpName,
  rmsNormEps,
  RUNTIME_SUPPORT,
  scalarParamValues,
  sliceAttrs,
  type UnaryOpName,
  WEIGHT_SLOTS,
  WHERE_OP,
} from "../ops.ts";
import { type ExecStep, type FusedStep, type FusionCounts, planFusions } from "./fusion.ts";
import {
  bindSymbols,
  countUses,
  declaredDtypes,
  eligibleCompressedInitializers,
  ExecutionError,
  type NodePlan,
  planGraph,
  type SymbolBindings,
  validateGraphContracts,
  weightChannelAxes,
} from "./plan.ts";
import {
  bakeBindGroups,
  type BindingRecipe,
  type BindingSource,
  derivePlanSlots,
  executeBakedPlan,
  executeStepRecipe,
  type OutputRecipe,
  type StepRecipe,
  StepRecipeBuilder,
  type TempSource,
  type ValueSource,
} from "./recipe.ts";

type TensorOf<D extends IrDtype, A> = {
  readonly dtype: D;
  /** 束縛解決済みの具体値。 */
  readonly shape: readonly number[];
  readonly data: A;
};

/**
 * 意味論 dtype で判別するテンソル（ADR 0009 — ADR 0008 の公開面の部分改訂）。
 *
 * 要素は全型 4 バイトで、**bool は u32 の 0 / 1**（WebGPU のストレージバッファに 1bit 型が
 * 無いため。GPU 側の格納と同じ規約）。入力・出力とも同じ形で扱う。
 */
export type Tensor =
  | TensorOf<"f32", Float32Array<ArrayBuffer>>
  | TensorOf<"i32", Int32Array<ArrayBuffer>>
  | TensorOf<"bool", Uint32Array<ArrayBuffer>>;

/** 意味論 dtype ごとのホスト側 TypedArray（診断と入力検査で使う）。 */
const HOST_ARRAY: Readonly<
  Record<IrDtype, Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor>
> = {
  f32: Float32Array,
  i32: Int32Array,
  bool: Uint32Array,
};

/** readback 先の Tensor を宣言 dtype から組み立てる（判別子と配列型を 1 箇所で対応づける）。 */
const hostTensor = (
  dtype: IrDtype,
  shape: readonly number[],
  buffer: ArrayBuffer,
  count: number,
): Tensor => {
  switch (dtype) {
    case "f32":
      return { dtype, shape, data: new Float32Array(buffer, 0, count) };
    case "i32":
      return { dtype, shape, data: new Int32Array(buffer, 0, count) };
    case "bool":
      return { dtype, shape, data: new Uint32Array(buffer, 0, count) };
  }
};

/** elementwise 族の生成入力のうち rank に依らない部分（rank はエンコード時に決まる）。 */
type ElementwiseOp =
  | { readonly op: UnaryOpName | BinaryOpName | typeof WHERE_OP; readonly dtype: IrDtype }
  | { readonly op: typeof CAST_OP; readonly dtype: IrDtype; readonly to: IrDtype };

export type RunInputs = Readonly<Record<string, Tensor>>;
export type RunOutputs = Readonly<Record<string, Tensor>>;

/**
 * 整数内積の変種（w8a8 経路）。**両者は同じ整数を返す**ので、これは速度の選択でしかない
 * （src/kernels/linear-i8a8.ts）。
 */
export type I8a8Dot = "dp4a" | "emu";

/**
 * **テスト専用の非公開面**（mod.ts からは輸出しない — ADR 0008 の「薄い面」を汚さない）。
 *
 * i8a8 の整数内積変種を強制する（linear / 融合 attention の**全 i8a8 カーネル共通** —
 * 選択は device の言語機能の列挙 1 つで決まるので Session 単位で 1 つ）。拡張のある機で
 * `dot4I8Packed` 版とエミュ版を実走して atol=0 で突合するのが「エミュは数値同一」という
 * 主張の唯一の機械的検出器で、環境変数ではなく Session 単位のノブにしてあるのは
 * 1 プロセス内で両方を回すため。
 */
export const I8A8_DOT: unique symbol = Symbol("karume.i8a8Dot");

/**
 * op 族ごとの計算精度ノブ（ADR 0028 / attention の i8a8 は設計 §9.2）。**重み格納の f16
 * （ADR 0018）とは別の軸**で、`"f16"` は共有タイルを f16 に落として内積を回す変種
 * （累積は f32）、`"i8a8"` は活性を per-token i8 へ量子化して整数内積で回す変種を選ぶ。
 *
 * MUST: 3 値は**相互排他**（直積ではない）。attention の q/k/v は全て活性で格納軸を持たない
 * ので、「f16 かつ i8a8」という組み合わせは表現する対象がそもそも存在しない。
 * MUST: `"f16"` は `acquireGpu({ shaderF16: true })` を伴う（Session 構築時に fail loudly）。
 * **`"i8a8"` は `shader-f16` を要求しない**（feature ゲートに混ぜないこと）。
 */
export type ComputePrecision = "f32" | "f16" | "i8a8";

export type SessionOptions = {
  /** submit の時間予算政策（TDR / watchdog 対策 — ADR 0004）。既定は DEFAULT_SUBMIT_POLICY。 */
  readonly submitPolicy?: SubmitPolicy;
  /**
   * linear の実行形（既定 `"f32"` = 従来どおり）。
   *
   * `"i8a8"` は **i8 で GPU 常駐している重みの linear** だけに効き、活性を per-token i8 へ
   * 量子化して整数内積で回す（設計 = docs/research/2026-08-03-dp4a-w8a8-design.md）。
   * `"f16"` は共有タイルを f16 に落とす計算変種（ADR 0028）で、重み格納が f32 / f16 の
   * linear に効く（**i8 常駐の重みとは組めない** — w8a16 は未実装なので fail loudly）。
   * MUST: 既定は `"f32"` — i8 / f16 資産を自動で低精度実行にすると既存の PNG sha256 門と
   * E2E tolerance が黙って変わる。opt-in 以外はあり得ない。
   */
  readonly linearCompute?: "f32" | "i8a8" | "f16";
  /**
   * 融合 attention（ADR 0023 の 3 カーネル）の実行形（既定 `"f32"` = 従来どおり）。
   *
   * `"f16"` は ①QK / ②行統計 / ③PV の共有タイルを f16 にし、**S も f16 で受け渡す**
   * （① が書き ②③ が読む — transient が半減する）。`linearCompute` と**別の軸**なのは、
   * 1024px の内訳が attention 46% / linear 42% で片方だけ f16 にしたい場面が実際にあるため。
   *
   * `"i8a8"` は q / k / v を i8 へ量子化して整数内積で回す変種
   * （設計 = docs/research/2026-08-04-attention-a8-design.md）。**現時点の意味論は
   * 「①QK と ③PV が i8a8・②行統計は f32 のまま」**（③ の A 側 = P̃ は scale が 1/127 に
   * 構造縮退するので量子化カーネルを通らず、V だけが Vᵀ 経由の per-column i8 になる）。
   * `"f16"` と違い `shader-f16` を要求せず、資産の格納形（f32 / f16 / i8）とも独立に効く —
   * attention の入力は全て活性だから。
   * **適格判定は段ごとに独立で、満たさない段だけが f32 経路へ沈黙で縮退する**
   * （① は `D % 4 == 0`・③ は `N % 4 == 0` — i8 ペイロードの語境界条件で、パック方向が
   * 段ごとに違う）。したがって **`D % 4 == 0` かつ `N % 4 != 0` なら「①QK は i8a8・③PV は
   * f32」の混成**になる。linear の `k % 4` と同じ流儀で、落ちたことは診断のパイプライン
   * キーにだけ出る。
   * MUST: 既定は `"f32"`。
   */
  readonly attentionCompute?: ComputePrecision;
  /**
   * 融合 attention の中間バッファ **S（スコア）の格納形**（既定 `"f32"` = 従来どおり）。
   * `attentionCompute`（計算形）と**直交する第 2 の軸**で、S は中間バッファなので
   * 「どの精度で計算するか」と「どの精度で置くか」を別々に選べる。
   *
   * `"f16"` は S を `array<u32>` に **`pack2x16float` で 2 要素／語**詰める（core WGSL・
   * **`shader-f16` を要求しない** — ADR 0030 決定 1「i8a8 は shader-f16 を要求しない」を
   * 保ったまま `attentionCompute: "i8a8"` と組める。本命はこの組）。丸めは格納の 1 回だけで、
   * 読み側の `unpack2x16float` は厳密。したがって出力は「**S をホストで f16 に丸めた
   * f32 変種**」とビット単位で一致する。
   *
   * MUST: 既定は `"f32"` — S の格納形を自動で落とすと既存の PNG sha256 門と E2E tolerance が
   * 黙って変わる（ADR 0028 決定 1 が auto を禁じたのと同じ理由）。
   * MUST: `attentionCompute: "f16"`（`:c16`）との併用は **fail loudly**。あちらは S を
   * `array<f16>` で持つ**別の格納形**なので、冗長かつ矛盾する組になる。
   * **適格判定は `D % 4 == 0 && N % 4 == 0`**（書き手 ①QK が v4 経路を取る条件 — 1 スレッドが
   * 4 連続列 = 2 語ちょうどを排他に書く）で、満たさない形は f32 格納へ**沈黙で**縮退する
   * （linear の `k % 4`・ADR 0030 決定 5 と同じ流儀で、落ちたことは診断のパイプライン
   * キーにだけ出る）。
   */
  readonly attentionScoreStorage?: ScoreStorage;
  /** テスト専用（{@link I8A8_DOT}）。既定は wgslLanguageFeatures の列挙から決める。 */
  readonly [I8A8_DOT]?: I8a8Dot;
};

/**
 * 低精度格納（f16 — ADR 0018 / i8 — ADR 0019）の実績。**ADR 0006 が義務づける常設診断**で、
 * 「f16 / i8 指定なのに適格 0MB」を沈黙させないための唯一の観測点。
 *
 * 対象は圧縮格納の initializer だけ（格納 f32 / i32 はどちらにも数えない）。両方 0 なら
 * 「そのモデルに低精度格納が 1 本も無い」、`resident` が 0 で `hostExpanded` が大きければ
 * 「低精度と宣言したのに適格判定で全部落ちている」— この 2 つが区別できる形にしてある。
 */
export type StorageDiagnostics = {
  /**
   * 圧縮のまま GPU 常駐した重みの **GPU バッファ上のバイト数**（整列のゼロ詰め込み。
   * i8 は **per-channel scale のバッファぶんも加算**する — 実際に GPU が抱えるバイト数を
   * 表す欄なので、scale を除くと VRAM 実績と食い違う）。
   * f32 で持ったときの 1/2（f16）・約 1/4（i8）になるのがこの経路の目的。
   */
  readonly residentCompressedBytes: number;
  /**
   * 適格外でロード時に CPU で f32 展開した重みの、**展開後**のバイト数（= 実際に GPU が
   * 抱えるバイト数）。VRAM 削減はゼロで、縮んだのは配信サイズだけ。
   */
  readonly hostExpandedBytes: number;
};

/**
 * params バッファの内容アドレスキャッシュの実績（1 run ぶん）。params は実行時のテンソル値に
 * 依存しないので、shape が変わらない限り 2 run 目以降の `allocCount` は 0 に落ちる。
 *
 * MUST: 常設診断として出す。キャッシュが外れても値は正しいまま（毎 dispatch 確保に戻るだけ）
 * で、例外も警告も出ない — ここが唯一の観測点。
 *
 * NOTE: 導出済み計画がヒットした run（{@link PreparedPlanStats}）では**導出相そのものが
 * 走らない**ため `allocCount` / `reuseCount` とも 0 になる。値の意味は変わらず、「その run が
 * params に対して行った GPU 操作がゼロ」という事実の報告。
 */
export type ParamsCacheStats = {
  /** この run で新規に確保 + writeBuffer した params の本数。 */
  readonly allocCount: number;
  /** この run でキャッシュから配り直した params の本数（GPU 操作ゼロ）。 */
  readonly reuseCount: number;
};

/**
 * 導出済み実行計画（Session 常駐）の実績。同一 bindings で走り直す run は計画・融合判定・
 * レシピ導出を丸ごと飛ばし、レシピ列をそのまま実行相へ渡す。
 *
 * MUST: 常設診断として出す。キャッシュが外れても値は正しいまま（毎 run 導出に戻るだけ）で、
 * 例外も警告も出ない — 性能だけが静かに戻る。ここが唯一の観測点。
 */
export type PreparedPlanStats = {
  /** この run が導出済み計画に当たったか。 */
  readonly hit: boolean;
  /** この run の決着時点で Session が抱えている導出済み計画の本数（上限あり）。 */
  readonly cachedPlans: number;
};

/**
 * transient slot の GPU backing（Session 常駐バッファ群）の実績。導出済み計画にヒットした run は
 * 中間バッファをここから配るので、アリーナの確保・参照計数・createBuffer / destroy がゼロになる。
 *
 * MUST: 常設診断として出す。**signature が交互に切り替わる形では毎 run 作り直しになり**、値は
 * 正しいまま run ごとに数百 MiB の createBuffer / destroy が復活する（例外も警告も出ない）。
 * `buildCount` が run 数に比例して伸びていないことが、その沈黙劣化の唯一の観測点。
 */
export type PlanBackingStats = {
  /**
   * 活性 backing が常駐させている **slot の**総バイト数（未構築 / 破棄済みなら 0）。
   * MUST: 定義は「slot 表の総バイト数」— backing が併せて常駐させる入力バッファは含めない
   * （理由と門は {@link ActiveBacking.bytes}）。
   */
  readonly residentBytes: number;
  /** Session の生存中に backing を構築した累計回数（run ごとではなく累計）。 */
  readonly buildCount: number;
};

export type SessionDiagnostics = {
  readonly pipelineCount: number;
  readonly submit: SubmitStats;
  /**
   * 重み（initializer）アリーナの実績。**params キャッシュ（Session 常駐）の実体もここが
   * 所有する**ので、`allocCount` は initializer 本数 + 生成済み params 本数になる。
   */
  readonly weights: ArenaStats;
  /** 低精度格納の適格 / 適格外の内訳（ADR 0006 の常設診断）。 */
  readonly storage: StorageDiagnostics;
  /**
   * 直近 run の中間バッファ実績。未実行なら undefined。
   *
   * NOTE: slot backing に乗った run（{@link PlanBackingStats}）では中間バッファも入力バッファも
   * アリーナを通らないため、ここに残るのは readback staging のぶんだけになる
   * （値の意味は不変 — 「その run がアリーナで確保したもの」）。
   */
  readonly lastRun: ArenaStats | undefined;
  /**
   * 直近 run の **op 別 GPU 実時間内訳**（パイプラインキー別 — ADR 0021）。
   * 計測が無効な device（`acquireGpu` の `gpuTiming` / feature 不在）では undefined。
   * `lastRun` と同じ寿命で、run の開始でリセットされる。
   */
  readonly lastRunTiming: GpuTimingStats | undefined;
  /**
   * 直近 run の**計画時**に適用が決まった融合 / 別名化の回数（ルール別 —
   * src/runtime/fusion.ts）。未実行なら undefined で、run のたびに丸ごと置き換わる。
   *
   * MUST: 常設診断として出す。融合はエクスポータのノード発行順が 1 つ変わるだけで黙って
   * 外れ、値は正しいまま性能だけが戻る（例外も警告も出ない）。ここが唯一の観測点。
   */
  readonly lastRunFusions: FusionCounts | undefined;
  /**
   * 直近 run の params キャッシュ実績。未実行なら undefined で、run のたびに置き換わる。
   * 実体（バッファ）は Session 常駐なので、ここは「その run が何本作り、何本使い回したか」。
   */
  readonly lastRunParams: ParamsCacheStats | undefined;
  /**
   * 直近 run の導出済み計画キャッシュ実績。run の開始でリセットされ、導出相が決着した時点で
   * 埋まる（未実行、および導出相の途中で落ちた run では undefined）。
   */
  readonly lastRunPrepared: PreparedPlanStats | undefined;
  /**
   * transient slot の GPU backing の実績（run ごとではなく Session の現況 + 累計）。
   * 未構築の Session では `{ residentBytes: 0, buildCount: 0 }`。
   */
  readonly planBacking: PlanBackingStats;
};

/** MUST: `queue.writeBuffer` で書くバッファはプール外（アリーナの不変条件）。 */
const HOST_WRITTEN_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
  GPUBufferUsage.COPY_SRC;
const PARAMS_STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const PARAMS_UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * 実行計画から解決済み shape を引く。plan.shapes は入力・initializer・全ノード出力を漏れなく
 * 持つ（planGraph の不変条件）ため、引けないのはランタイム内部の不変条件破れ。
 * MUST: 空 shape（スカラ）へ縮退させない — 要素数 1 として素通りし、確保サイズも要素数検査も
 * 静かに誤ったまま実行が進む。
 */
const resolvedShape = (
  shapes: ReadonlyMap<string, readonly number[]>,
  name: string,
): readonly number[] => {
  const shape = shapes.get(name);
  if (shape === undefined) {
    throw new ExecutionError(`値 '${name}' の解決済み shape が実行計画に無い`);
  }
  return shape;
};

/**
 * i8 格納の per-channel scale テンソル（ADR 0019）。実在・F32・broadcast 可能形は
 * openModel（format/container.ts）が済ませているので、ここは view を組むだけ。
 *
 * MUST: `Float32Array` の view はコピーせずに張る（scale は重み本体に比べれば小さいが、
 * ここで無条件コピーを挟むと「生バイトのまま常駐」の経路が二重確保になる）。絶対 offset の
 * 4 バイト整列は safetensors リーダが保証済み。
 */
const scaleTensor = (
  file: SafetensorsFile,
  name: string,
  scaleKey: string | undefined,
): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
} => {
  if (scaleKey === undefined) {
    throw new ExecutionError(`initializer '${name}': 格納 i8 なのに storage.scale が無い`);
  }
  const view = file.tensors.get(scaleKey);
  if (view === undefined) {
    throw new ExecutionError(`initializer '${name}': scale テンソル '${scaleKey}' が無い`);
  }
  return {
    bytes: tensorBytes(file, view),
    values: new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4),
    shape: view.shape,
  };
};

/**
 * GPU 常駐経路の scale が**平坦添字で引ける形**であることを見る（ADR 0019）。
 *
 * カーネルは `wscale[出力チャネル]` と読む。したがって scale はチャネル軸だけが伸びた
 * keepdim 形（`[Cout,1,1]` 等）でなければならない。broadcast 可能なだけの形（例: 重み
 * `[1,5]` に対する `[1,5]`）は openModel を通ってしまうが、カーネルは先頭要素しか読まない
 * ため**沈黙誤値**になる — 適格経路ではここが唯一の門。
 */
const assertChannelScale = (
  name: string,
  weightShape: readonly number[],
  scaleShape: readonly number[],
  axis: number | undefined,
): void => {
  if (axis === undefined) {
    throw new ExecutionError(`initializer '${name}': per-channel scale のチャネル軸が決まらない`);
  }
  const ok = scaleShape.length === weightShape.length &&
    scaleShape.every((dim, index) => dim === (index === axis ? weightShape[axis] : 1));
  if (!ok) {
    throw new ExecutionError(
      `initializer '${name}': scale [${scaleShape.join(",")}] が重み [${
        weightShape.join(",")
      }] の軸 ${axis} の keepdim 形でない`,
    );
  }
};

/**
 * 導出相まるごとの成果物（Session 常駐 — キーは {@link Session.#preparedKey}）。
 *
 * MUST: 後段が実際に参照するものだけを持つ（`GraphPlan.nodes` / `GraphPlan.bindings` は
 * レシピ導出が終われば誰も読まない）。読まれない導出物を抱えると、キャッシュの寿命が
 * 「run の入出力に効く事実」から離れ、何を再利用しているのかが読めなくなる。
 */
type PreparedPlan = {
  /** 入力・initializer・全ノード出力の解決済み shape（`#uploadInput` / `#readOutputs` が読む）。 */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly recipes: readonly StepRecipe[];
  /** 計画時に決まった融合回数（ヒット run もこの値を常設診断へ報告する — ADR 0040 §3）。 */
  readonly fusions: FusionCounts;
};

/**
 * 導出相の前半（計画 → 融合判定）だけの成果物。後半（レシピ導出）は params の writeBuffer と
 * パイプライン生成を伴い run の errorScope 区間の内側でしか出せないので、そこまではステップ列
 * のまま持ち越す。MUST: {@link PreparedPlan} と `recipes` の有無で判別する（`in`）。
 */
type PlannedSteps = {
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly fusions: FusionCounts;
  readonly steps: readonly ExecStep[];
};

/**
 * 導出済み計画の常駐本数の上限（LRU）。
 *
 * 定数で固定するのは、これが「連続する run が同じ bindings を使い回す」局所性だけを拾う
 * 器で、増やしても効かない形（run ごとに shape が変わる）では 1 本目から効かないため —
 * 設定ノブにすると当たらないキャッシュを太らせる調整に化ける。
 */
const PREPARED_PLAN_CAPACITY = 4;

/**
 * 活性 signature の transient slot backing（**容量 1**）。
 *
 * 容量 1 なのは、slot は run の中間バッファそのもの（DiT で ~1GiB 規模）で、signature ごとに
 * 抱えると VRAM が本数倍になるため — 導出済み計画（ホストのオブジェクトだけ）を 4 本持てる
 * のとは前提が違う。
 */
type ActiveBacking = {
  /** この backing が属する導出済み計画のキー（{@link Session.#preparedKey}）。 */
  readonly key: string;
  /**
   * transient slot の常駐バイト数（**入力バッファは含まない**）。
   * MUST: この定義を変えない — 「slot 表の総バイト数 = 非 backed run のプール確保」という
   * footprint 不変の門（tests/gpu_plan_backing_test.ts）がこの値そのもので、入力ぶんを混ぜると
   * 門が観測しているものが変わる（入力は signature ごとに固定サイズで、slot に比べれば桁が違う）。
   */
  readonly bytes: number;
  /**
   * グラフ入力名 → backing 所有の常駐バッファ（`HOST_WRITTEN_USAGE`）。サイズは解決済み
   * shape から確定するので、同一 signature の run では作り直す理由が無い。
   */
  readonly inputs: ReadonlyMap<string, GPUBuffer>;
  /** 焼き込み済み bind group（{@link bakeBindGroups}）。backed run はこれを dispatch するだけ。 */
  readonly groups: readonly (readonly GPUBindGroup[])[];
  /** グラフ出力名 → 実体（読み戻し先 — 構築時に確定。backed run は env を組まない）。 */
  readonly outputs: ReadonlyMap<string, GPUBuffer>;
  /** backing が所有する全バッファ（slot + 入力 — 破棄と readback 適格判定の分岐に使う）。 */
  readonly owned: ReadonlySet<GPUBuffer>;
  /**
   * readback を許す唯一の集合 = pin された slot + 入力バッファ。
   * MUST: 入力を含める — グラフ出力が入力の別名になる形（reshape(入力)）では読み戻し先が
   * 入力バッファそのものになる。非 backed 経路の入力はプール外（`allocHostWritten`）で
   * `RunArena.isReadable` が素通しにしていたので、これはその同値の再現。
   */
  readonly readable: ReadonlySet<GPUBuffer>;
};

type SessionState = {
  readonly gpu: GpuContext;
  readonly model: KarumeModel;
  readonly cache: PipelineCache;
  readonly scheduler: SubmitScheduler;
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  /**
   * params バッファの内容アドレスキャッシュ（キー = usage + 全要素の連結 —
   * {@link Session.#writeParams}）。実体は weights アリーナが所有する Session 常駐バッファで、
   * ここは「内容 → 既に上げてあるバッファ」の索引だけを持つ。
   * MUST: モジュールスコープに置かない（副作用ゼロの不変条件 — Session ごとに device も
   * バッファも別）。
   */
  readonly paramsCache: Map<string, GPUBuffer>;
  /**
   * 解決済み bindings → 導出済み実行計画（LRU・上限 {@link PREPARED_PLAN_CAPACITY}）。
   * MUST: モジュールスコープに置かない（副作用ゼロの不変条件 — Session ごとに graph も
   * 常駐バッファも別で、レシピはその実体を直参照で畳み込んでいる）。
   */
  readonly prepared: Map<string, PreparedPlan>;
  /**
   * 重みスロットの格納形（圧縮のまま常駐した initializer だけが載る）。ここに無い値は
   * f32 として読む — カーネル変種の選択はこの表 1 つで決まる。
   */
  readonly weightStorages: ReadonlyMap<string, WeightStorage>;
  /** i8 で常駐した重みの per-channel scale バッファ（変種の追加束縛 — ADR 0019）。 */
  readonly weightScaleBuffers: ReadonlyMap<string, GPUBuffer>;
  readonly storage: StorageDiagnostics;
  /** linear の実行形（opt-in — {@link SessionOptions.linearCompute}）。 */
  readonly linearCompute: "f32" | "i8a8" | "f16";
  /** 融合 attention の実行形（opt-in — {@link SessionOptions.attentionCompute}）。 */
  readonly attentionCompute: ComputePrecision;
  /** S の格納形（opt-in — {@link SessionOptions.attentionScoreStorage}）。計算形と直交する軸。 */
  readonly attentionScoreStorage: ScoreStorage;
  /**
   * i8a8 の整数内積変種。既定は `navigator.gpu.wgslLanguageFeatures` の列挙から決まり、
   * テストは {@link I8A8_DOT} で強制できる。**どちらでも数値は 1 ビットも変わらない**。
   */
  readonly i8a8Dot: I8a8Dot;
  readonly useCounts: ReadonlyMap<string, number>;
  readonly dtypes: ReadonlyMap<string, IrDtype>;
  readonly outputNames: ReadonlySet<string>;
};

export class Session {
  readonly #state: SessionState;
  #lastRun: ArenaStats | undefined;
  #lastRunFusions: FusionCounts | undefined;
  #lastRunParams: ParamsCacheStats | undefined;
  #lastRunPrepared: PreparedPlanStats | undefined;
  /** 進行中 run の params 実績（run の頭でリセットし、決着時に {@link #lastRunParams} へ移す）。 */
  #paramsAllocCount = 0;
  #paramsReuseCount = 0;
  /** 活性 signature の slot backing（容量 1 — {@link ActiveBacking}）。 */
  #backing: ActiveBacking | undefined;
  #backingBuilds = 0;
  /**
   * 破棄待ちの slot バッファ。切替・追い出し・dispose はここへ積むだけで、実際の `destroy()` は
   * **flush 後の 1 箇所**（{@link Session.#destroyRetired}）でだけ起きる。
   */
  readonly #retired: GPUBuffer[] = [];
  /**
   * 実行中 / 待機中の run と dispose の直列化チェーン。決着（成功・失敗）だけを次に渡すため
   * 自身は決して reject しない。
   */
  #chain: Promise<void> = Promise.resolve();
  #disposal: Promise<void> | undefined;

  private constructor(state: SessionState) {
    this.#state = state;
  }

  /** MUST: 構築の入口はここだけ（重みアップロードを含む明示 async ステージ）。 */
  static async create(
    gpu: GpuContext,
    model: KarumeModel,
    options: SessionOptions = {},
  ): Promise<Session> {
    // 非対応 op / 格納 dtype は全件列挙して落とす（1 件ずつ落とすと何本足りないか分からない）
    assertRuntimeSupport(model.graph, RUNTIME_SUPPORT);
    validateGraphContracts(model.graph);
    const linearCompute = options.linearCompute ?? "f32";
    const attentionCompute = options.attentionCompute ?? "f32";
    const attentionScoreStorage = options.attentionScoreStorage ?? "f32";
    // MUST: S の格納形は 1 つに決まらなければならない。`:c16` は S を array<f16> で持つ
    // **別の形**（ADR 0028）なので、s16 と併記されたら黙ってどちらかに解釈せず落とす
    // （どちらの丸め列で走ったのかが診断からも数値からも見えなくなる）。
    if (attentionScoreStorage === "f16" && attentionCompute === "f16") {
      throw new ExecutionError(
        "attentionScoreStorage 'f16' と attentionCompute 'f16' は同時に指定できない" +
          "（attentionCompute 'f16' は S を array<f16> で持つ別の格納形 — " +
          "shader-f16 無しで S を半分にするなら attentionCompute を 'f32' か 'i8a8' にすること）",
      );
    }
    // MUST: f16 計算を要求されたのに feature が無い device なら**ここで落とす**。黙って f32
    // 経路へ落とすと、既定経路と opt-in の区別が診断からも数値からも見えなくなる
    // （ADR 0025 決定 1 と同じ理由）。
    if ((linearCompute === "f16" || attentionCompute === "f16") && !gpu.shaderF16Enabled) {
      throw new ExecutionError(
        "f16 計算変種を要求したが、device が 'shader-f16' を有効化していない" +
          `（linearCompute: ${linearCompute} / attentionCompute: ${attentionCompute}）。` +
          "acquireGpu({ shaderF16: true }) を渡して device を取り直すこと" +
          "（feature は device 作成時にしか要求できない）",
      );
    }

    const scheduler = new SubmitScheduler(gpu, options.submitPolicy);
    const weights = new RunArena(gpu.device, () => scheduler.flush());
    const weightBuffers = new Map<string, GPUBuffer>();
    // 圧縮格納のまま上げてよい initializer（消費が重みスロットだけ — ADR 0018）。
    // グラフ構造だけで決まるので契約検査の直後に 1 度求めれば足りる。
    const eligible = eligibleCompressedInitializers(model.graph);
    // i8 の per-channel scale が掛かる軸（消費側 op から決まる — ADR 0019）。
    const channelAxes = weightChannelAxes(model.graph);
    const weightStorages = new Map<string, WeightStorage>();
    const weightScaleBuffers = new Map<string, GPUBuffer>();
    let residentCompressedBytes = 0;
    let hostExpandedBytes = 0;
    // MUST: 重みアップロードも errorScope で囲む（ADR 0004 の「errorScope 常設」）。上限超過の
    // createBuffer は同期例外を投げずに無効バッファを返し、無効バッファ / 整列違反への
    // writeBuffer も警告すら出さない no-op になるため、包まないと重みが空のまま走り出す。
    // MUST NOT: このループの中で await しない。push から pop の発行までを 1 つの同期区間に
    // 保つことが、device 単位ロックを取らずに LIFO の交錯を防いでいる根拠になっている。
    pushFailureScopes(gpu.device);
    try {
      for (const [name, initializer] of Object.entries(model.graph.initializers)) {
        const view = model.file.tensors.get(initializer.tensor);
        if (view === undefined) {
          throw new ExecutionError(
            `initializer '${name}': テンソル '${initializer.tensor}' が無い`,
          );
        }
        const raw = tensorBytes(model.file, view);
        // 格納 f16 / i8 だけが 2 経路に分かれる（ADR 0018 / 0019）。適格なら生バイトのまま
        // 常駐させ dequant はカーネル内（VRAM 削減はこれで初めて成立する）、適格外はここで
        // f32 へ展開する（正しさは保たれ VRAM 削減はゼロ）。他の格納 dtype は生バイトが
        // そのまま GPU 表現。
        let payload: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer> = raw;
        if (initializer.storage.dtype === "f16") {
          if (eligible.has(name)) {
            // MUST: 奇数要素長は末尾 2 バイトのゼロ詰めで 4 バイト整列させる。writeBuffer は
            // 4 の倍数でないサイズを validation で拒む（= 重みが空のまま走り出す）。
            payload = alignF16Payload(raw);
            weightStorages.set(name, "f16");
            residentCompressedBytes += payload.byteLength;
          } else {
            payload = decodeF16(raw);
            hostExpandedBytes += payload.byteLength;
          }
        }
        if (initializer.storage.dtype === "i8") {
          const scale = scaleTensor(model.file, name, initializer.storage.scale);
          // initializer の宣言 shape は数値のみ（parseIrGraph が保証 — 記号次元は拒否）。
          const shape = model.graph.values[name].shape.map(Number);
          if (eligible.has(name)) {
            assertChannelScale(name, shape, scale.shape, channelAxes.get(name));
            // MUST: 要素数が 4 の倍数でない重みは末尾をゼロ詰めして 4 バイト整列させる
            // （f16 の 2 バイト詰めと同じ理由 — writeBuffer が validation で落ちる）。
            payload = alignI8Payload(raw);
            weightStorages.set(name, "i8");
            // MUST: scale のバッファも「GPU 常駐圧縮」に数える（実際に抱えるバイト数）。
            residentCompressedBytes += payload.byteLength + scale.bytes.byteLength;
            const scaleBuffer = weights.allocHostWritten(
              Math.max(4, scale.bytes.byteLength),
              HOST_WRITTEN_USAGE,
            );
            if (scale.bytes.byteLength > 0) {
              gpu.device.queue.writeBuffer(scaleBuffer, 0, scale.bytes);
            }
            weightScaleBuffers.set(name, scaleBuffer);
          } else {
            payload = decodeI8(raw, shape, scale.values, scale.shape);
            hostExpandedBytes += payload.byteLength;
          }
        }
        const buffer = weights.allocHostWritten(
          Math.max(4, payload.byteLength),
          HOST_WRITTEN_USAGE,
        );
        if (payload.byteLength > 0) gpu.device.queue.writeBuffer(buffer, 0, payload);
        weightBuffers.set(name, buffer);
      }
    } catch (cause) {
      // MUST: 後始末の失敗で本体の例外を上書きしない（run 側と同じ規律）。原因は本体側に
      // あり、destroy の rejection（主因は device 消失）に差し替わると調査の起点が消える。
      // push した 2 本は必ず pop して積み残さない。
      await discardFailureScopes(gpu.device);
      await weights.destroy().catch(() => undefined);
      throw cause;
    }
    const failure = await popFailureScopes(gpu.device, "重みのアップロード");
    if (failure !== undefined) {
      await weights.destroy().catch(() => undefined);
      throw failure;
    }

    // MUST: Session を返す前に**実際の submit を 1 回**出して完了まで待つ。queue.writeBuffer は
    // staging を確保して溜め込み、submit の完了までそれを解放しない — 数 GiB の重みを上げた
    // 直後は VRAM が二重計上のまま最初の run に入り、初回ピークが重み 1 本ぶん押し上がる
    // （f16 preset で実測 +2.7GiB。docs/research/2026-08-08-vram-oom-misreport.md §4）。
    // MUST NOT: scheduler.flush() で代用しない。pending dispatch が空だと submit を出さずに
    // 即 return するため、staging は溜まったまま残る。
    // NOTE: submit ごとの onSubmittedWorkDone を禁じているのは run のホットパス（submit.ts の
    // 「計測の帰属」）で、ここは Session 生成ごと 1 回・窓の外なので推定にも壁時計にも乗らない。
    // NOTE: errorScope で囲まないのは、空の submit が確保も検証も伴わないため（両建てで囲む
    // のは「確保を伴う区間」— device.ts の pushFailureScopes）。加えて Session.create は
    // GpuContext のスコープロック外なので、await を跨ぐスコープをここに張ると並行 Session の
    // 失敗を誤帰属させる口になる。
    try {
      gpu.device.queue.submit([]);
      // MUST: device 消失時 onSubmittedWorkDone は解決しない（ハングにしないため競わせる）。
      await gpu.raceDeviceLost(gpu.device.queue.onSubmittedWorkDone(), "重みのアップロード");
    } catch (cause) {
      await weights.destroy().catch(() => undefined);
      throw cause;
    }

    return new Session({
      gpu,
      model,
      cache: new PipelineCache(gpu.device),
      scheduler,
      weights,
      weightBuffers,
      paramsCache: new Map(),
      prepared: new Map(),
      weightStorages,
      weightScaleBuffers,
      storage: { residentCompressedBytes, hostExpandedBytes },
      linearCompute,
      attentionCompute,
      attentionScoreStorage,
      // 拡張の有無は**速度にしか効かない**（両変種は同じ整数を返す）ので、機能検出ではなく
      // 経路選択としてここで 1 度だけ決める（src/kernels/linear-i8a8.ts の docstring）。
      i8a8Dot: options[I8A8_DOT] ??
        (dp4aAvailable(gpu.wgslLanguageFeatures) ? "dp4a" : "emu"),
      useCounts: countUses(model.graph),
      dtypes: declaredDtypes(model.graph),
      outputNames: new Set(model.graph.outputs),
    });
  }

  /**
   * 入力を束縛してグラフを実行し、**グラフ出力のみ**読み戻す。
   * `bindings` は記号次元の明示指定で、入力 shape から導いた束縛と食い違えば fail loudly。
   *
   * MUST: 同一 Session の run は呼び出し順に**直列化**される（await せず並行発行しても実行は
   * 1 本ずつ）。errorScope は device 単位の LIFO スタックで、run は await を跨いでスコープを
   * 張り続けるため、重なると自分の validation 失敗を null として取り逃がし、無関係な run が
   * 代わりに落ちる。先行 run の失敗は後続 run に伝播しない。
   * NOTE: 同一 device 上の**別 Session** との重なりは、この直列化では防げない（Session ごとに
   * 別のチェーンになる）。そちらは GpuContext の errorScope 区間ロックが受け持つ。
   */
  run(inputs: RunInputs, bindings: SymbolBindings = {}): Promise<RunOutputs> {
    // dispose 済みの判定は呼び出し時点で行う（チェーンの中で見ると、dispose より前に発行した
    // run まで巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      return Promise.reject(new ExecutionError("dispose 済みの Session では実行できない"));
    }
    return this.#enqueue(() => this.#runOnce(inputs, bindings));
  }

  /** 重みバッファを解放する。実行中の run の完了を待ってから破棄し、以後の run は fail loudly。 */
  dispose(): Promise<void> {
    // MUST: 2 度目以降も同じ完了を返す。先に返すと呼び出し側が「破棄済み」と見なして
    // device.destroy() まで進み、flush-before-destroy が崩れる。
    // MUST: slot backing の破棄も**この 1 本に相乗り**させる（破棄経路の担い手を増やさない —
    // ADR 0004）。weights.destroy() は flush の完了まで待つので、その後の destroy は
    // flush-before-destroy を満たす。失敗（主因は device 消失）しても破棄は必ず行う。
    this.#disposal ??= this.#enqueue(async () => {
      try {
        await this.#state.weights.destroy();
      } finally {
        this.#retireBacking();
        this.#destroyRetired();
      }
    });
    return this.#disposal;
  }

  diagnostics(): SessionDiagnostics {
    return {
      pipelineCount: this.#state.cache.size,
      submit: this.#state.scheduler.stats,
      weights: this.#state.weights.stats,
      storage: this.#state.storage,
      lastRun: this.#lastRun,
      lastRunTiming: this.#state.scheduler.timing,
      lastRunFusions: this.#lastRunFusions,
      lastRunParams: this.#lastRunParams,
      lastRunPrepared: this.#lastRunPrepared,
      planBacking: {
        residentBytes: this.#backing?.bytes ?? 0,
        buildCount: this.#backingBuilds,
      },
    };
  }

  /** 直前の実行の決着後に `body` を走らせる。戻り値はこの呼び出し自身の結果 / 例外。 */
  #enqueue<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#chain.then(body);
    this.#chain = result.then(() => undefined, () => undefined);
    return result;
  }

  async #runOnce(inputs: RunInputs, bindings: SymbolBindings): Promise<RunOutputs> {
    const { gpu, model, scheduler } = this.#state;
    const graph = model.graph;

    // MUST: 利用者の inputs 由来のキーを蓄積する器は null プロトタイプ（理由は plan.ts の
    // bindSymbols と同じ）。素の `{}` では入力名が "__proto__" のとき shape 配列が
    // [[Prototype]] に化け、own property が作られないまま「入力が渡されていない」で落ちる。
    const inputShapes: Record<string, readonly number[]> = Object.create(null);
    for (const [name, tensor] of Object.entries(inputs)) inputShapes[name] = tensor.shape;
    // MUST: 束縛の解決（= 入力 shape の検証）はヒット・ミスに関わらず**毎 run 走らせる**。
    // ここが飛ぶと、キャッシュに当たった run だけ入力 shape の宣言不一致を素通りする。
    const resolved = bindSymbols(graph, inputShapes, bindings);
    const preparedKey = this.#preparedKey(resolved);
    const prepared = this.#takePrepared(preparedKey);
    // 計画（planGraph）と融合判定（planFusions）はどちらも GPU に触れない純関数で、ヒット時は
    // 丸ごと飛ばす（根拠は {@link Session.#preparedKey}）。融合は掴めなかったノードを素のまま
    // ステップ列に並べるので、この段は「速くなるか」だけを決め、正しさには関与しない。
    const derived = prepared ?? this.#planSteps(resolved);
    const shapes = derived.shapes;
    // MUST: ヒット run も融合回数を報告する（ADR 0040 §3 の常設契約 — キャッシュの有無で
    // 観測点が消えると、融合が外れた状態がヒット run の裏に隠れる）。
    this.#lastRunFusions = derived.fusions;
    this.#lastRunPrepared = undefined;
    this.#paramsAllocCount = 0;
    this.#paramsReuseCount = 0;

    const device = gpu.device;
    const arena = new RunArena(device, () => scheduler.flush());
    // MUST: env は **run 寿命の実体だけ**（グラフ入力とノード出力）。Session 常駐の重み /
    // per-channel scale は導出相で直参照へ畳むので、run ごとに写す必要が無い — 写すと
    // 「run の器に Session 常駐の状態が混ざる」形が残り、レシピの再利用が成り立たなくなる。
    const env = new Map<string, GPUBuffer>();
    // MUST: **run が GPU 操作を発行するのは自分のロック区間内のみ**（エンコード〜flush〜pop〜
    // readback〜アリーナ破棄まで）。errorScope は device 単位の LIFO で、操作の失敗は発行時点
    // のスタック先頭に帰属するため、「スコープを張らない操作だからロック外でよい」は成り立た
    // ない — 張らない操作の失敗こそ他人のスコープに入る。Session 内は #chain で直列化済みだ
    // が、同一 device 上の**別 Session** の run とは重なりうる（GpuContext の不変条件を参照）。
    // MUST NOT: この区間の内側からロックを再取得しない（自己デッドロック — 内側の層は同期
    // 区間で完結するスコープのみを使う）。
    return await gpu.withScopeLock(async () => {
      // GPU 時間内訳の寿命は **直近 run**（ADR 0021）。ロックを取ってから捨てることで、
      // 表が「今から積む run のぶんだけ」になる（先行 run の回収は既に済んでいる）。
      scheduler.resetTiming();
      let outputs: RunOutputs;
      // 活性化した slot backing（ミス run では undefined のまま）。readback の適格判定が
      // 「その実体が pin された slot か」を見るため、エンコード区間の外まで持ち越す。
      let backing: ActiveBacking | undefined;
      // この run が backing を新規構築したか（失敗時の回復規律 — 下の catch が読む）。
      let builtBacking = false;
      try {
        // 無効な bindGroup / dispatch は throw せず submit ごと失敗し、出力にプール残骸が
        // 残る沈黙故障になる。中間バッファの確保も同じ区間で out-of-memory を見る。
        pushFailureScopes(device);
        let popped = false;
        try {
          // 導出相 → 実行相。どちらも run の errorScope 区間の内側で、間に submit を挟まない
          // （params の writeBuffer は導出相で出るので、それを読む dispatch の submit より
          // 必ず先に発行される）。ヒット run はこの導出相ごと飛ぶ — params の writeBuffer も
          // パイプライン生成も出ないので、GPU 操作はレシピ実行のぶんだけになる。
          let recipes: readonly StepRecipe[];
          if ("recipes" in derived) {
            recipes = derived.recipes;
            // MUST: 入力の検査（値依存 — 毎 run）は backing 構築より前。ここで落ちる run に
            // slot（DiT で ~GiB 規模）の構築を払わせない。
            const data = graph.inputs.map((spec) =>
              this.#checkInput(spec.name, inputs[spec.name], shapes)
            );
            // MUST: backing を作るのは**ヒット run だけ**。単発 run（1 回しか走らない
            // ワークロード）に slot メモリを払わせないための唯一の門で、ミス run の挙動と
            // ArenaStats はこれで完全に据え置かれる。
            const activated = this.#activateBacking(preparedKey, recipes, shapes);
            backing = activated.backing;
            builtBacking = activated.built;
            graph.inputs.forEach((spec, index) => {
              this.#writeInput(activated.backing, spec.name, data[index]);
            });
          } else {
            for (const spec of graph.inputs) {
              env.set(spec.name, this.#uploadInput(spec.name, inputs[spec.name], shapes, arena));
            }
            recipes = await this.#buildRecipes(derived.steps);
            // MUST: 登録は `#buildRecipes` が**完走して戻った後**だけ。途中で throw した run の
            // 部分レシピを載せると、次の同一 bindings の run が欠けたステップ列を実行し、
            // 例外なしで誤った値を返す。
            this.#registerPrepared(preparedKey, { shapes, recipes, fusions: derived.fusions });
          }
          this.#lastRunPrepared = {
            hit: prepared !== undefined,
            cachedPlans: this.#state.prepared.size,
          };
          if (backing === undefined) {
            const run = { device, scheduler, arena, env };
            for (const recipe of recipes) executeStepRecipe(recipe, run);
            arena.assertDrained();
          } else {
            // slot 経路。積むコマンド列はアリーナ経路と同一で、bind 先の実体が run を跨いで
            // 固定されるだけ（前 run の残骸が残っていてよい根拠は full-write — ADR 0014）。
            // bind group は構築時に焼き込み済みなので、ここは dispatch を積むだけ。
            executeBakedPlan(recipes, backing.groups, scheduler);
          }

          await scheduler.flush();
          const pending = popFailureScopes(device, "run のエンコード");
          popped = true;
          const failure = await pending;
          if (failure !== undefined) throw failure;
        } catch (cause) {
          // MUST: 失敗した run の残 pending dispatch は submit せずに捨てる（エンコード途中の
          // throw で先行ノードのぶんが残る）。実行する理由が無いうえ、後始末の
          // arena.destroy() → flush で出すと、破棄されるバッファを参照したまま submit する
          // ことになる。discard は同期なので、捨てるまでの間に submit の隙が生まれない。
          scheduler.discard();
          if (!popped) await discardFailureScopes(device);
          throw cause;
        }

        // backed run は env を組まない（読み戻し先は焼き込み時に確定した写像）。
        outputs = await this.#readOutputs(backing?.outputs ?? env, shapes, arena, backing);
        this.#lastRun = arena.stats;
        this.#lastRunParams = {
          allocCount: this.#paramsAllocCount,
          reuseCount: this.#paramsReuseCount,
        };
      } catch (cause) {
        // MUST: 後始末の失敗で本体の例外を上書きしない。原因は本体側にあり、destroy の
        // rejection（主因は device 消失）に差し替わると調査の起点が消える。
        await arena.destroy().catch(() => undefined);
        // MUST: **この run が新規構築した** backing だけを退役させる。一過性の失敗（構築時の
        // out-of-memory 等）は次の同一 signature ヒットでの再構築で回復し、壊れたまま常駐した
        // backing が後続 run に居座らない。既存 backing での失敗 run は退役させない — 無関係な
        // 失敗のたびに ~GiB 規模の再構築を強いるスラッシングになる（そちらの回復手段は
        // signature 切替と LRU 追い出し）。
        if (builtBacking) this.#retireBacking();
        // MUST: 破棄待ちの slot は arena.destroy（= flush / discard 済み）の**後**に返す。
        this.#destroyRetired();
        throw cause;
      }
      // 本体が通ったときは後始末の失敗をそのまま伝える（flush 未完了のまま返さない）。
      await arena.destroy();
      // 切替 / 追い出しで浮いた旧 backing を返す唯一の後始末点（flush 後 — ADR 0004 の
      // flush-before-destroy）。dispose が同じことをするので、ここを通らずに落ちた run の
      // 積み残しも取りこぼさない。
      this.#destroyRetired();
      return outputs;
    });
  }

  /** 導出相の前半（計画 → 融合判定）。GPU に触れない純関数だけで閉じる。 */
  #planSteps(bindings: SymbolBindings): PlannedSteps {
    const plan = planGraph(this.#state.model.graph, bindings);
    const fusion = planFusions(plan.nodes, {
      useCounts: this.#state.useCounts,
      outputNames: this.#state.outputNames,
    });
    return { shapes: plan.shapes, fusions: fusion.counts, steps: fusion.steps };
  }

  /**
   * 導出済み計画のキー — 解決済み bindings を `graph.symbols` の**宣言順**に並べた値の連結。
   * シンボルを持たないグラフはキー `""` の 1 本になる。
   *
   * MUST: これが「導出相を丸ごと飛ばしてよい」根拠の全て。graph は Session 構築時に固定で、
   * 計画・融合判定・レシピ導出は graph と bindings だけの純関数だから、**キーが解決済み
   * bindings の完全一致なら導出結果は構造的に同一**で、planGraph の契約検査群（宣言 shape
   * との照合・dtype 解決・未束縛シンボル）も同じ判定に落ちる。したがってヒット run が
   * 検査を飛ばしても fail loudly が緩むことはない（同じ入力に対する同じ検査の再実行だけを
   * 省く）。入力 shape 自体の検証は {@link bindSymbols} が毎 run 行う。
   * MUST: 全シンボルが束縛済みであることは bindSymbols が保証する（未束縛なら例外）ので、
   * ここで欠けを気にしなくてよい。
   */
  #preparedKey(bindings: SymbolBindings): string {
    return this.#state.model.graph.symbols.map((sym) => bindings[sym]).join(",");
  }

  /** キャッシュを引き、当たったら最近使用へ回す（Map の挿入順が LRU の順序そのもの）。 */
  #takePrepared(key: string): PreparedPlan | undefined {
    const cached = this.#state.prepared.get(key);
    if (cached === undefined) return undefined;
    this.#state.prepared.delete(key);
    this.#state.prepared.set(key, cached);
    return cached;
  }

  /**
   * 導出済み計画を載せ、上限を超えたら最も古いものを落とす。
   *
   * NOTE: 追い出しで捨てるのは**ホスト側のオブジェクトだけ**で、GPU 資源の破棄は伴わない。
   * レシピが直参照している実体（params バッファは paramsCache が持つ weights アリーナ、
   * pipeline / layout は {@link PipelineCache}）はいずれも Session 常駐で、寿命は
   * `Session.dispose` に一本化されている（ADR 0004「確保と破棄を 1 箇所へ」）。ここで
   * destroy すると、同じ実体を指す別の計画や次の導出が破棄済みバッファを掴む。
   */
  #registerPrepared(key: string, plan: PreparedPlan): void {
    this.#state.prepared.set(key, plan);
    // 1 run につき 1 本しか増えないので、超過は常にちょうど 1 本。
    if (this.#state.prepared.size > PREPARED_PLAN_CAPACITY) {
      const oldest = this.#state.prepared.keys().next().value;
      if (oldest !== undefined) {
        this.#state.prepared.delete(oldest);
        // MUST: 追い出された計画の slot backing は宙に浮く（次にその bindings が来ても
        // ミス run になり backing は使われない）。持ち続けると、二度と当たらない signature の
        // 中間バッファぶんの VRAM を Session の寿命いっぱい抱え込む。
        if (this.#backing?.key === oldest) this.#retireBacking();
      }
    }
  }

  /**
   * ヒット run の slot backing を活性化する（無ければ構築・別 signature なら作り直す）。
   * `built` は「この run が新規構築したか」— 失敗時の回復規律（{@link Session.#runOnce}）が読む。
   *
   * MUST: 呼ぶのは run の `withScopeLock` / errorScope 区間の内側だけ。createBuffer は上限超過で
   * 同期例外を投げずに無効バッファを返し、createBindGroup の validation 失敗も例外にならない
   * ため、囲まないと「無効な slot / bind group に dispatch が書く」沈黙故障になる。
   * NOTE: レシピ列は必ず 1 度アリーナ経路で完走している（backing はヒット run でしか作らない）
   * ので、参照計数が閉じることは既に `assertDrained` が確かめ済み — slot 導出はその同じ簿記を
   * 仮想的に再生するだけで、新しい正しさの前提を持ち込まない。
   */
  #activateBacking(
    key: string,
    recipes: readonly StepRecipe[],
    shapes: ReadonlyMap<string, readonly number[]>,
  ): { readonly backing: ActiveBacking; readonly built: boolean } {
    const current = this.#backing;
    if (current !== undefined && current.key === key) return { backing: current, built: false };
    // MUST: 旧 backing は destroy せず破棄待ちへ積む（この run の flush 後にだけ返す）。
    this.#retireBacking();
    const graph = this.#state.model.graph;
    const device = this.#state.gpu.device;
    const slots = derivePlanSlots(recipes);
    const buffers = slots.bytes.map((size) => device.createBuffer({ size, usage: STORAGE_USAGE }));
    // 入力バッファも backing 所有にする（run ごとの確保と writeBuffer 先の入れ替わりを消す）。
    // MUST: 大きさはアリーナ経路の `#uploadInput` と同じ算式（4 バイト床込み — 0 要素入力で
    // 0 サイズバッファを束縛しない）。同一 signature なら不変。
    const inputs = new Map<string, GPUBuffer>(
      graph.inputs.map((spec) => [
        spec.name,
        device.createBuffer({
          size: Math.max(4, numel(resolvedShape(shapes, spec.name)) * 4),
          usage: HOST_WRITTEN_USAGE,
        }),
      ]),
    );
    const baked = bakeBindGroups(recipes, slots, { device, buffers, inputs });
    // 読み戻し先はここで確定する。initializer がグラフ出力になる形（IR が許す）は値名の
    // 写像に載らないので、`#readOutputs` 側の重みフォールバックがそのまま受け持つ。
    const outputs = new Map<string, GPUBuffer>();
    for (const name of graph.outputs) {
      const buffer = baked.values.get(name);
      if (buffer !== undefined) outputs.set(name, buffer);
    }
    const backing: ActiveBacking = {
      key,
      bytes: slots.bytes.reduce((total, size) => total + size, 0),
      inputs,
      groups: baked.groups,
      outputs,
      owned: new Set([...buffers, ...inputs.values()]),
      readable: new Set([
        ...[...slots.pinned].map((slot) => buffers[slot]),
        ...inputs.values(),
      ]),
    };
    this.#backing = backing;
    this.#backingBuilds += 1;
    return { backing, built: true };
  }

  /** 活性 backing を破棄待ちへ移す（実際の `destroy()` は flush 後の後始末点で 1 回だけ）。 */
  #retireBacking(): void {
    const current = this.#backing;
    if (current === undefined) return;
    for (const buffer of current.owned) this.#retired.push(buffer);
    this.#backing = undefined;
  }

  /**
   * 破棄待ちの slot バッファを実際に破棄する。
   *
   * MUST: 呼ぶのは flush（または discard）の完了後だけ — 破棄済みバッファを参照する
   * エンコードを submit すると、コマンドバッファ丸ごと失敗して無関係な dispatch まで
   * 実行されないまま誤った値が静かに残る（ADR 0004）。
   * NOTE: device 消失後の `destroy()` は無害な no-op。
   */
  #destroyRetired(): void {
    for (const buffer of this.#retired) buffer.destroy();
    this.#retired.length = 0;
  }

  /**
   * このノードの重みスロットの格納形（カーネル変種の選択 — ADR 0018）。
   *
   * MUST: スロット位置は適格判定と**同じ表**（{@link WEIGHT_SLOTS}）から引く。片方だけ
   * ずれると、圧縮のまま上げた重みを f32 カーネルが読む（あるいはその逆）ビット列の
   * 読み替えになり、例外は 1 つも出ない。
   */
  #weightStorage(step: NodePlan): WeightStorage {
    const slot = WEIGHT_SLOTS.get(step.node.op);
    if (slot === undefined) {
      throw new ExecutionError(`op '${step.node.op}' に重みスロットの定義が無い`);
    }
    return this.#state.weightStorages.get(step.node.ins[slot]) ?? "f32";
  }

  /**
   * i8 変種の追加束縛（per-channel scale）。f32 / f16 では空配列になり、bind group は従来の
   * ままになる（ADR 0019）。
   *
   * MUST: 束縛番号はカーネル側の定数（`*_SCALE_BINDING`）から引く。WGSL の宣言と executor が
   * 別々に番号を持つと、変種を足したときに片方だけずれる。
   */
  #weightScaleBindings(
    step: NodePlan,
    storage: WeightStorage,
    binding: number,
  ): readonly BindingRecipe[] {
    if (storage !== "i8") return [];
    const slot = WEIGHT_SLOTS.get(step.node.op);
    if (slot === undefined) {
      throw new ExecutionError(`op '${step.node.op}' に重みスロットの定義が無い`);
    }
    const name = step.node.ins[slot];
    const buffer = this.#state.weightScaleBuffers.get(name);
    if (buffer === undefined) {
      throw new ExecutionError(`initializer '${name}': i8 常駐なのに scale バッファが無い`);
    }
    return [{ binding, source: { kind: "resident", buffer } }];
  }

  /** 値の宣言 dtype。引けないのはランタイム内部の不変条件破れ（declaredDtypes は全値を持つ）。 */
  #declaredDtype(name: string): IrDtype {
    const dtype = this.#state.dtypes.get(name);
    if (dtype === undefined) {
      throw new ExecutionError(`値 '${name}' の宣言 dtype が無い`);
    }
    return dtype;
  }

  /**
   * 入力テンソルの検査（**値依存なので毎 run 必要** — ヒット run でも飛ばせない）。通れば
   * GPU へ書くホスト配列をそのまま返す。
   */
  #checkInput(
    name: string,
    tensor: Tensor | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): Tensor["data"] {
    // 未指定・shape 不一致は bindSymbols が済ませている。ここは dtype と要素数だけを見る。
    if (tensor === undefined) throw new ExecutionError(`入力 '${name}' が渡されていない`);
    // MUST: 判別子と実データの両方を宣言 dtype と突き合わせる。要素は全型 4 バイトなので、
    // 取り違えは writeBuffer を素通りしてビット列の読み替えになる（例外は出ない）。
    const dtype = this.#declaredDtype(name);
    const expected = HOST_ARRAY[dtype];
    if (tensor.dtype !== dtype || !(tensor.data instanceof expected)) {
      throw new ExecutionError(
        `入力 '${name}': 宣言 dtype '${dtype}'（${expected.name}）に対し渡されたのは '${tensor.dtype}' / ${tensor.data.constructor.name}`,
      );
    }
    const shape = resolvedShape(shapes, name);
    const count = numel(shape);
    if (tensor.data.length !== count) {
      throw new ExecutionError(
        `入力 '${name}': 要素数 ${tensor.data.length} が shape [${
          shape.join(",")
        }] の ${count} と合わない`,
      );
    }
    return tensor.data;
  }

  /** アリーナ経路の入力アップロード（run 寿命のバッファを 1 本確保して書く）。 */
  #uploadInput(
    name: string,
    tensor: Tensor | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
  ): GPUBuffer {
    const data = this.#checkInput(name, tensor, shapes);
    const buffer = arena.allocHostWritten(Math.max(4, data.length * 4), HOST_WRITTEN_USAGE);
    if (data.length > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /**
   * backed run の入力書き込み（backing 所有の常駐バッファへ**上書き**する）。
   *
   * MUST: この `writeBuffer` が追い越せる未 submit エンコードは存在しない。根拠は 2 つ —
   * ①同一 Session の run は {@link Session.#chain} で直列化され、各 run は flush
   * （`onSubmittedWorkDone`）と readback の完了まで済ませてからしか返らないので、呼ばれた時点で
   * 先行 run の未 submit エンコードは 1 つも残っていない ②run の中では入力書き込みが全
   * エンコードに先行する。`queue.writeBuffer` は未 submit の先行エンコードを追い越す
   * （ADR 0004 不変条件④）ため、どちらかが崩れると前 run の dispatch が新しい入力を読む
   * 沈黙誤値になる。
   */
  #writeInput(backing: ActiveBacking, name: string, data: Tensor["data"]): void {
    const buffer = backing.inputs.get(name);
    if (buffer === undefined) throw new ExecutionError(`入力 '${name}' の常駐バッファが無い`);
    if (data.length > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, data);
  }

  /**
   * 導出相 — ステップ列をレシピ列へ落とす。GPU コマンドを 1 つも出さず、run 寿命の実体
   * （{@link RunArena} のバッファ）にも触れない（モジュール doc の MUST）。
   */
  async #buildRecipes(steps: readonly ExecStep[]): Promise<readonly StepRecipe[]> {
    // 実体は実行相まで決まらないので、導出相は「その値名が既に定義済みか」だけを追う
    // （束縛漏れを実行相へ持ち越さず、ここで fail loudly にする）。
    const defined = new Set(this.#state.model.graph.inputs.map((spec) => spec.name));
    const recipes: StepRecipe[] = [];
    for (const step of steps) recipes.push(await this.#buildStep(step, defined));
    return recipes;
  }

  /**
   * 値名 → bind 面の出どころ。重みは Session 常駐なので実体まで解決し、それ以外
   * （グラフ入力・ノード出力・別名）は値名のまま残す。ノード内一時にはなりえないので
   * 戻りは {@link ValueSource}（別名元としてもそのまま使える）。
   */
  #bindingSource(name: string, defined: ReadonlySet<string>): ValueSource {
    const resident = this.#state.weightBuffers.get(name);
    if (resident !== undefined) return { kind: "resident", buffer: resident };
    if (!defined.has(name)) throw new ExecutionError(`値 '${name}' のバッファが無い`);
    return { kind: "value", name };
  }

  /**
   * 実行ステップ 1 つ（素のノード または 融合ステップ）のレシピ。
   *
   * MUST: 確保 → retain → 本体 → 入力の release（延べ）→ 定義ぶんの release、という簿記は
   * 両者で**1 本**に閉じる（実行側は {@link executeStepRecipe}）。融合ごとに手書きの解放簿記を
   * 置くと、アリーナの参照計数が融合の本数だけ別実装になり、1 本でもずれると例外なしの
   * 沈黙誤値になる（早すぎる解放ならプール再利用で値が化け、多すぎれば peak が落ちない）。
   */
  async #buildStep(step: ExecStep, defined: Set<string>): Promise<StepRecipe> {
    const outputName = step.kind === "node" ? step.plan.outputName : step.outputName;
    const outputShape = step.kind === "node" ? step.plan.outputShape : step.outputShape;
    // bind 面のオペランド順（重複無し）と解放簿記の延べ列は別物。素のノードでは
    // どちらも node.ins に一致し、融合ステップだけが 2 つを別々に宣言する。
    const bindNames = step.kind === "node" ? step.plan.node.ins : step.binds;
    const consumedNames = step.kind === "node" ? step.plan.node.ins : step.ins;
    const binds = bindNames.map((name) => this.#bindingSource(name, defined));
    // reshape と恒等 expand は要素順を変えないので**入力バッファをそのまま出力の実体にする**
    // （別名 — ADR 0011）。dispatch も確保も出さない。要素数一致は planGraph が済ませているので、
    // 別名先の実バッファは宣言 shape ぶんの大きさを必ず満たす。
    // MUST: 別名元は bind 面の先頭（temp にはなりえない）。
    const output: OutputRecipe = step.kind === "node" && step.aliasesInput
      ? { kind: "alias", source: binds[0] }
      : { kind: "alloc", byteLength: numel(outputShape) * 4 };
    // 出力は「この値名」として束ねる。実行相が dispatch より前に env へ載せるので、
    // 同一ステップ内の bind もこの 1 本で解決できる。
    const out: BindingSource = { kind: "value", name: outputName };

    const builder = new StepRecipeBuilder();
    if (step.kind === "node") {
      await this.#buildNode(step.plan, step.aliasesInput, binds, out, builder);
    } else {
      await this.#buildFused(step, binds, out, builder);
    }
    defined.add(outputName);

    return {
      outputName,
      output,
      uses: this.#state.useCounts.get(outputName) ?? 0,
      pinned: this.#state.outputNames.has(outputName),
      temps: builder.temps,
      dispatches: builder.dispatches,
      releases: consumedNames,
    };
  }

  /**
   * 融合ステップの 1 dispatch。bind 面は「params, 入力…, 出力」で全ルール共通、params は
   * 16 バイトの uniform で固定（src/runtime/fusion.ts の {@link FusedDispatch}）。
   */
  async #buildFused(
    step: FusedStep,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const { key, gridItems, workgroupSize } = step.dispatch;
    const { pipeline, layout } = await this.#state.cache.get(key, step.dispatch.wgsl());
    const params = this.#writeParams(step.dispatch.params, PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      gridItems,
      workgroupSize,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: binds.length + 1, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /** 素のノード 1 つの本体（確保・retain・解放は {@link executeStepRecipe} が済ませる）。 */
  async #buildNode(
    step: NodePlan,
    aliasesInput: boolean,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    switch (step.contract.kind) {
      case "unary":
      case "binary":
        await this.#buildElementwise(
          step,
          { op: step.contract.name, dtype: step.inputDtypes[0] },
          binds,
          out,
          builder,
        );
        break;
      case "cast":
        await this.#buildElementwise(
          step,
          { op: CAST_OP, dtype: step.inputDtypes[0], to: step.outputDtype },
          binds,
          out,
          builder,
        );
        break;
      case "where":
        // 生成入力は**値スロット**の dtype（スロット 0 は条件で常に bool）。
        await this.#buildElementwise(
          step,
          { op: WHERE_OP, dtype: step.inputDtypes[1] },
          binds,
          out,
          builder,
        );
        break;
      case "cumsum":
        await this.#buildCumsum(step, binds, out, builder);
        break;
      case "matmul":
        await this.#buildMatmul(step, binds, out, builder);
        break;
      case "bmm":
        await this.#buildBmm(step, binds, out, builder);
        break;
      case "gather":
        await this.#buildGather(step, binds, out, builder);
        break;
      case "rowReduce":
        await this.#buildRowReduce(step, step.contract.name, binds, out, builder);
        break;
      case "permute":
      case "slice":
      case "symPrefixSlice":
        await this.#buildStridedCopy(step, step.contract.kind, binds, out, builder);
        break;
      case "expand":
        // 恒等 expand は別名化済み（0 dispatch）。複製軸が 1 つでもあれば実体化コピー。
        if (!aliasesInput) {
          await this.#buildStridedCopy(step, step.contract.kind, binds, out, builder);
        }
        break;
      case "cat":
        await this.#buildCat(step, binds, out, builder);
        break;
      case "pad":
        await this.#buildPad(step, binds, out, builder);
        break;
      case "flip":
        await this.#buildFlip(step, binds, out, builder);
        break;
      case "linear":
        await this.#buildLinear(step, binds, out, builder);
        break;
      case "layerNorm":
        await this.#buildLayerNorm(step, binds, out, builder);
        break;
      case "rmsNorm":
        await this.#buildRmsNorm(step, binds, out, builder);
        break;
      case "softmax":
        await this.#buildSoftmax(step, binds, out, builder);
        break;
      case "attention":
        await this.#buildAttention(step, binds, out, builder);
        break;
      case "embedding":
        await this.#buildEmbedding(step, binds, out, builder);
        break;
      case "maskedFill":
        await this.#buildMaskedFill(step, binds, out, builder);
        break;
      case "conv1d":
        await this.#buildConv1d(step, binds, out, builder);
        break;
      case "conv2d":
        await this.#buildConv2d(step, binds, out, builder);
        break;
      case "convTranspose1d":
        await this.#buildConvTranspose1d(step, binds, out, builder);
        break;
      case "reshape":
        // 別名化は #buildStep で済んでいる（この op は 1 dispatch も出さない）。
        break;
      default: {
        // MUST: 未処理の kind をここで止める。switch が素通りすると出力バッファに 1 バイトも
        // 書かれないまま次のノードへ進み、プール再利用の残骸がそのまま値になる
        // （full-write 不変条件の破れ — ADR 0014）。型としては到達不能で、op を足したときの
        // 結線漏れをコンパイル時に赤くするのがこの分岐の役目。
        const unhandled: never = step.contract;
        throw new ExecutionError(`未処理の op kind: ${JSON.stringify(unhandled)}`);
      }
    }
  }

  async #buildElementwise(
    step: NodePlan,
    element: ElementwiseOp,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    // rank 0（スカラ）は codegen の rank ≥ 1 契約に合わせて長さ 1 の 1 次元に正規化する。
    const outShape = step.outputShape.length === 0 ? [1] : [...step.outputShape];
    const rank = outShape.length;
    const spec: ElementwiseSpec = element.op === CAST_OP
      ? { op: CAST_OP, rank, dtype: element.dtype, to: element.to }
      : { op: element.op, rank, dtype: element.dtype };
    const key = elementwiseKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(key, elementwiseWgsl(spec));
    // attrs のスカラ（clamp の min/max など）は params の末尾に f32 で載る（並びは契約表）。
    const params = this.#writeParams(
      elementwiseParams(
        outShape,
        step.inputShapes,
        scalarParamValues(step.contract, step.node.attrs, `nodes (${step.node.op})`),
      ),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      ELEMENTWISE_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: binds.length + 1, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * reduce 族（sum / amax / amin）。**縮約軸で 2 変種に踏み分ける**:
   *
   * - 最終次元 → 行 reduce（既存カーネル・1 行 = 1 workgroup + 256 幅ツリー）
   * - それ以外 → 軸 reduce（1 スレッド = 1 出力・コアレス読み。縮約順序は行 reduce と
   *   厳密に一致 = 出力ビット同一 — src/codegen/reduce.ts の {@link axisReduceWgsl}）
   *
   * MUST: 分岐は**軸だけ**で決める。速度で選ぶ余地を作ると、最終次元でも軸変種が走る形が
   * でき、既定経路のビット不変（PNG sha256 門）が実行時条件に依存してしまう。
   */
  async #buildRowReduce(
    step: NodePlan,
    op: ReduceOpName,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const inputShape = step.inputShapes[0];
    const axis = reduceDim(step.node.attrs, `nodes (${step.node.op})`);
    // 要素型はキーに載る（bool の sum は u32 を読んで i32 の個数を書く — ADR 0009）。
    const spec = { op, dtype: step.inputDtypes[0] };
    const lastDim = axis === inputShape.length - 1;
    const outCount = numel(step.outputShape);
    const key = lastDim ? reduceKey(spec) : axisReduceKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      lastDim ? reduceWgsl(spec) : axisReduceWgsl(spec),
    );
    const inner = numel(inputShape.slice(axis + 1));
    const params = this.#writeParams(
      lastDim
        ? reduceParams(outCount, inputShape[axis])
        : axisReduceParams(outCount, inputShape[axis], inner),
      PARAMS_UNIFORM_USAGE,
    );
    // 行 reduce は 1 行 = 1 workgroup、軸 reduce は 1 スレッド = 1 出力。どちらも上限を
    // 超えたら縮退させ、カーネル側の grid-stride で回す。
    const groups = gridStrideWorkgroups(
      outCount,
      lastDim ? 1 : AXIS_REDUCE_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * cumsum（最終次元の前縁和）。**1 invocation = 1 行**の逐次走査で、行方向を grid-stride で
   * 回す（形の根拠は src/kernels/cumsum.ts）。
   */
  async #buildCumsum(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const { pipeline, layout } = await this.#state.cache.get(CUMSUM_KEY, CUMSUM_WGSL);
    const params = this.#writeParams(cumsumParams(rows, dim), PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      rows,
      CUMSUM_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: CUMSUM_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * permute / 非恒等 expand / slice / sym_prefix_slice の実体化コピー（strided 読み 1 カーネル族 —
   * ADR 0011 / 0010 / 0014）。出力は常に連続で、入力側だけを stride で読む。expand の複製軸は
   * stride 0、sym_prefix_slice は **Tmax 形の入力**の連続 stride、slice は入力の連続 stride と
   * **開始位置 offset**。恒等 expand は別名化されるのでここへは来ない。
   */
  async #buildStridedCopy(
    step: NodePlan,
    kind: "permute" | "expand" | "slice" | "symPrefixSlice",
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const where = `nodes (${step.node.op})`;
    const srcStrides = kind === "permute"
      ? permuteSrcStrides(srcShape, permuteDims(step.node.attrs, where))
      : kind === "expand"
      ? expandSrcStrides(srcShape, outShape)
      : kind === "slice"
      ? sliceSrcStrides(srcShape)
      // MUST: 束縛後の outShape ではなく **srcShape（Tmax 形）** から組む（prefixSliceSrcStrides
      // の MUST — 送り幅を縮めると 2 行目以降が別の行を読む）。
      : prefixSliceSrcStrides(srcShape);
    // offset は ADR 0011 の (offset, strides[4]) モデルそのもの。permute / expand /
    // sym_prefix_slice は入力の先頭から読むので 0 で、**slice だけが 0 以外**を取る
    // （ADR 0011 が予告した「可変点 1 語」がここ）。
    let offset = 0;
    if (kind === "slice") {
      const { dim, start } = sliceAttrs(step.node.attrs, where);
      offset = sliceSrcOffset(srcShape, dim, start);
    }
    // 要素型はキーに載る（bool マスク / i32 添字の expand が実測にある — ADR 0009）。
    const spec = { dtype: step.outputDtype };
    const key = stridedKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(key, stridedWgsl(spec));
    const params = this.#writeParams(
      stridedParams(outShape, srcStrides, offset),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      STRIDED_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * cat（strided 書きコピー族 — ADR 0014）。入力ごとに **1 dispatch** で出力の部分領域へ書く。
   * 入力は連続で読み、出力へ `(offset, 出力の連続 strides)` で書く（読み族の双対）。
   *
   * MUST: 書き出し位置の総和が連結軸の長さとちょうど一致することを確かめる（full-write）。
   * 一致しなければ出力のどこかが**未書き込みのまま**残り、プール再利用なら前の値がそのまま
   * 結果になる。契約の shape 規則（軸長 = 入力の軸長の総和）と同じ事実をエンコード側の
   * 積み上げからも確かめる形で、片方だけの誤りを内部矛盾として落とす。
   */
  async #buildCat(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const outShape = step.outputShape;
    const where = `nodes (${step.node.op})`;
    const dim = catDim(step.node.attrs, where);
    const outStrides = catOutStrides(outShape);
    const spec = { dtype: step.outputDtype };
    const key = stridedWriteKey(spec);
    const { pipeline, layout } = await this.#state.cache.get(key, stridedWriteWgsl(spec));
    let written = 0;
    for (const [index, source] of binds.entries()) {
      const srcShape = step.inputShapes[index];
      const params = this.#writeParams(
        stridedWriteParams(srcShape, outStrides, catOutOffset(outShape, dim, written)),
        PARAMS_STORAGE_USAGE,
      );
      const groups = gridStrideWorkgroups(
        numel(srcShape),
        STRIDED_WRITE_WORKGROUP_SIZE,
        this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
      );
      builder.dispatch({
        key,
        pipeline,
        layout,
        params,
        bindings: [{ binding: 1, source }, { binding: 2, source: out }],
        workgroups: [groups, 1, 1],
      });
      written += srcShape[dim];
    }
    if (written !== outShape[dim]) {
      throw new ExecutionError(
        `${where}: cat の書き込みが出力全域を覆わない（軸 ${dim} に ${written} / 出力は ${
          outShape[dim]
        }）`,
      );
    }
  }

  /**
   * pad（最終次元の定数 0 埋め）。**1 dispatch で出力の全バイトを書く**（範囲内は転写・
   * 範囲外は 0 — ADR 0014 の full-write）。ゼロ初期化されたバッファを前提にしない。
   */
  async #buildPad(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const { left, right } = padAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(PAD_KEY, PAD_WGSL);
    const params = this.#writeParams(
      padParams(numel(srcShape.slice(0, -1)), srcShape[srcShape.length - 1], left, right),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      PAD_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: PAD_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * flip（静的軸の添字反転）。軸の位置は `[outer, len, inner]` の 3 分割に畳んで渡す
   * （rank を params に載せない — src/kernels/flip.ts）。
   */
  async #buildFlip(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = flipDim(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(FLIP_KEY, FLIP_WGSL);
    const params = this.#writeParams(
      // MUST: 軸の前後で分ける（`slice(0, dim)` と `slice(dim + 1)`）。境界を 1 つずらすと
      // 反転する軸が隣にずれ、shape も要素数も変わらないまま値だけが誤る。
      flipParams(numel(shape.slice(0, dim)), shape[dim], numel(shape.slice(dim + 1))),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(shape),
      FLIP_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: FLIP_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  async #buildMatmul(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [a, b] = step.inputShapes;
    const [m, k] = a;
    const n = b[1];
    // v4（vec4 の読み書き）は形状から導く 1 ビット。導出時に評価してキーと WGSL の
    // 両方へ渡す（同一キー ⇔ 同一バイト列は保たれる）。
    const v4 = gemmUsesVec4(k, n);
    // MUST: タイル幾何は行数 M のバケット（src/kernels/gemm-geometry.ts）。キー・WGSL・
    // dispatch の 3 つに**同じ m** を通す — 1 つでも渡し忘れると出力タイルが静かに欠ける。
    const key = matmulKey(v4, m);
    const { pipeline, layout } = await this.#state.cache.get(key, matmulWgsl(v4, m));
    const params = this.#writeParams(matmulParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `matmul [${a.join(",")}] × [${b.join(",")}]`;
    const geometry = gemmGeometryForRows(m);
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        1,
      ],
    });
  }

  /**
   * バッチ matmul（rank-3）。タイル 2 軸に加えて**バッチを z 軸**へ載せる。
   * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、3 軸とも上限超過は fail loudly。
   */
  async #buildBmm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [a, b] = step.inputShapes;
    const [batch, m, k] = a;
    const n = b[2];
    const v4 = gemmUsesVec4(k, n);
    // 幾何のバケットは**行列 1 枚の m**（バッチは z 軸で、タイル幾何とは独立）。
    const key = bmmKey(v4, m);
    const { pipeline, layout } = await this.#state.cache.get(key, bmmWgsl(v4, m));
    const params = this.#writeParams(bmmParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `bmm [${a.join(",")}] × [${b.join(",")}]`;
    const geometry = gemmGeometryForRows(m);
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        // バッチは 1 workgroup = 1 バッチ。ここも縮退させるとバッチが丸ごと未書き込みになる。
        tiledWorkgroups(batch, 1, limit, where),
      ],
    });
  }

  /**
   * 最終次元の gather。出力は連続で、`row = i / J` から `src[row * D + index[i]]` を引く。
   * 範囲外添字の扱いは src/kernels/gather.ts の裁定（GPU は NaN 汚染 / CPU 参照は throw）。
   */
  async #buildGather(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const count = numel(outShape);
    const { pipeline, layout } = await this.#state.cache.get(GATHER_KEY, GATHER_WGSL);
    const params = this.#writeParams(
      gatherParams(count, outShape[outShape.length - 1], srcShape[srcShape.length - 1]),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      count,
      GATHER_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: GATHER_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * linear（融合 op — ADR 0012）。先行次元を平坦化して `[m,k] × [k,n] + bias` の 2 次元
   * GEMM に落とす。重みは `[n,k]` の転置レイアウトのままカーネルが読む（転置コピー無し）。
   * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、上限超過は fail loudly。
   */
  async #buildLinear(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const n = weight[0];
    const k = weight[1];
    const m = numel(x.slice(0, -1));
    const weightStorage = this.#weightStorage(step);
    // w8a8 は **opt-in × i8 常駐 × k % 4 == 0** の 3 条件が揃ったときだけ（ADR 0025 予定）。
    // 既定の "f32" では 1 バイトも挙動が変わらない。
    if (this.#state.linearCompute === "i8a8" && weightStorage === "i8" && k % 4 === 0) {
      await this.#buildLinearI8a8(step, binds, out, builder, m, n, k);
      return;
    }
    // f16 計算変種（ADR 0028）。**i8 常駐の重みとは組めない**（w8a16 は未実装）ので、
    // 黙って f32 へ落とさずここで落とす — 落とすと「i8 の層だけ f16 が効かない」形になり、
    // 診断のキーを見ない限り気づけない。
    const compute: GemmCompute = this.#state.linearCompute === "f16" ? "f16" : "f32";
    if (compute === "f16" && weightStorage === "i8") {
      throw new ExecutionError(
        `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
          "linearCompute 'f16' は i8 常駐の重みとは組めない（w8a16 は未実装 — ADR 0028）。" +
          "linearCompute を 'i8a8' にするか、この重みを f32 / f16 格納で持つこと",
      );
    }
    const v4 = gemmUsesVec4(k, n);
    // MUST: タイル幾何は平坦化後の行数 m のバケット（src/kernels/gemm-geometry.ts）。
    // キー・WGSL・dispatch に**同じ m** を通す。
    const key = linearKey(weightStorage, v4, compute, m);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      linearWgsl(weightStorage, v4, compute, m),
    );
    const params = this.#writeParams(linearParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `linear [${x.join(",")}] × [${weight.join(",")}]`;
    const geometry = gemmGeometryForRows(m);
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, LINEAR_SCALE_BINDING),
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        1,
      ],
    });
  }

  /**
   * linear の **w8a8 変種**（opt-in — {@link SessionOptions.linearCompute}）。**1 ノード =
   * 2 dispatch**（融合 attention と同じ「複数 dispatch で 1 ノード」の扱い）:
   *
   * ① `quantize_rows`（活性を per-token i8 へ・行方向 grid-stride）→ ② i8a8 GEMM（整数内積・
   * 1 workgroup = 1 出力タイルなので上限超過は fail loudly）。
   *
   * MUST: 一時バッファ（`xq` / `xs`）は宣言 → ノード末尾で解放する。これで実行相の参照計数が
   * 閉じ（`assertDrained`）、失敗経路でも `arena.destroy()` が拾う。
   * MUST: `k` のオーバフロー門は fail loudly。黙って通すと i32 の巻き戻りで符号ごと化ける。
   */
  async #buildLinearI8a8(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
    m: number,
    n: number,
    k: number,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const where = `linear i8a8 [${x.join(",")}] × [${weight.join(",")}]`;
    if (k > LINEAR_I8A8_MAX_K) {
      throw new ExecutionError(
        `${where}: k=${k} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
          "（linearCompute を 'f32' にするか、この linear を i8 常駐から外す）",
      );
    }
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // 量子化した活性 `xq`（i8 を 4 詰め）と per-token scale `xs`。ノード内で閉じた一時領域。
    const xq = builder.allocTemp(Math.max(4, m * (k / 4) * 4));
    const xs = builder.allocTemp(Math.max(4, m * 4));

    // ① 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await this.#state.cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    builder.dispatch({
      key: QUANTIZE_ROWS_KEY,
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      params: this.#writeParams(quantizeRowsParams(m, k), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: xq },
        { binding: 3, source: xs },
      ],
      workgroups: [gridStrideWorkgroups(m, 1, limit), 1, 1],
    });

    // ② 整数内積の GEMM。タイル幾何は op → 幾何の純関数が決める（src/kernels/i8a8-geometry.ts）
    // — キーに載るので「同一キー → バイト同一 WGSL」は保たれる。
    const v4 = linearI8a8UsesVec4(n);
    const geometry = defaultI8a8Geometry("linear");
    const key = linearI8a8Key(v4, this.#state.i8a8Dot === "dp4a", geometry);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      linearI8a8Wgsl(v4, this.#state.i8a8Dot === "dp4a", geometry),
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params: this.#writeParams(linearI8a8Params(m, n, k), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: xq },
        { binding: 2, source: binds[1] },
        { binding: 3, source: binds[2] },
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, "i8", LINEAR_SCALE_BINDING),
        { binding: LINEAR_ACT_SCALE_BINDING, source: xs },
      ],
      workgroups: [
        tiledWorkgroups(n, i8a8TileN(geometry), limit, where),
        tiledWorkgroups(m, i8a8TileM(geometry), limit, where),
        1,
      ],
    });

    // MUST: ノード境界で一時バッファを返す（アリーナの不変条件）。
    builder.releaseTemp(xs);
    builder.releaseTemp(xq);
  }

  /** layer_norm（最終次元・affine あり）。1 行 = 1 workgroup で、行方向は grid-stride。 */
  async #buildLayerNorm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const { eps } = layerNormAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(LAYER_NORM_KEY, LAYER_NORM_WGSL);
    const params = this.#writeParams(
      layerNormParams(rows, dim, eps),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: LAYER_NORM_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * rms_norm（最終次元・weight のみ）。layer_norm と同じ 1 行 = 1 workgroup の形で、
   * 縮約は二乗和 1 パス（ADR 0017）。
   */
  async #buildRmsNorm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const eps = rmsNormEps(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(RMS_NORM_KEY, RMS_NORM_WGSL);
    const params = this.#writeParams(rmsNormParams(rows, dim, eps), PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: RMS_NORM_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 3, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /** softmax（最終次元、safe-softmax）。layer_norm と同じ 1 行 = 1 workgroup の形。 */
  async #buildSoftmax(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const { pipeline, layout } = await this.#state.cache.get(SOFTMAX_KEY, SOFTMAX_WGSL);
    const params = this.#writeParams(softmaxParams(rows, dim), PARAMS_UNIFORM_USAGE);
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: SOFTMAX_KEY,
      pipeline,
      layout,
      params,
      bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: out }],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * 融合 attention（ADR 0023）。**1 ノード = 3 dispatch**（`cat` と同じ「複数 dispatch で
   * 1 ノード」の扱い。full-write は**ノードの出力 O について**成立する）:
   *
   * ① QK gemm（S を実体化・scale はタイル充填時に q/k 両方へ）→ ② 行統計（m と 1/Σexp）→
   * ③ PV gemm（A タイル充填時に `exp(S−m)·inv` を評価 = P 非実体化）。
   *
   * 省略可能な第 4 入力 `mask[1,1,M,N]`（加算型）は **① の束縛が 1 本増えるだけ**で、
   * dispatch 数も ②③ の経路も変わらない（S が mask 済みで出てくる）。
   *
   * i8a8 変種（opt-in）では ① が 3 dispatch・③ が 3 dispatch に増える（最大 7）。**適格判定は
   * 段ごとに独立**（① は `D % 4 == 0`・③ は `N % 4 == 0`）なので、片方だけ i8a8 の**混成**が
   * 起こりうる — 満たさない段だけが f32 経路へ**沈黙で**縮退する（linear の `k % 4` と同じ
   * 流儀で、落ちたことは診断のパイプラインキーにだけ出る）。
   *
   * MUST: ① と ③ は 1 workgroup = 1 タイルなので、3 軸とも上限超過は fail loudly
   * （{@link tiledWorkgroups}）。縮退させるとタイルが欠落し、例外なしに O の一部が
   * 未書き込み（プール再利用なら前の値）で残る。② だけが行方向 grid-stride。
   * MUST: 一時バッファ（S / 行統計）は宣言 → ノード末尾で解放する。これで実行相の参照計数が
   * 閉じ（`assertDrained`）、失敗経路でも `arena.destroy()` が拾う（確保と破棄を 1 箇所へ —
   * ADR 0004）。
   */
  async #buildAttention(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [q, k, v] = step.inputShapes;
    // B と H は 1 本のバッチ軸へ畳む（契約は rank-4 head-first で、3 者の B / H は一致）。
    const batch = q[0] * q[1];
    const rows = q[2];
    const cols = k[2];
    const depth = q[3];
    const scale = attentionScale(step.node.attrs, `nodes (${step.node.op})`);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `attention [${q.join(",")}] × [${k.join(",")}] × [${v.join(",")}]`;

    // f16 計算変種（ADR 0028）。3 カーネルが**同時に**切り替わる — S の格納形が ① の書き手と
    // ②③ の読み手で一致していなければならないので、段ごとの混成はあり得ない。
    const compute: GemmCompute = this.#state.attentionCompute === "f16" ? "f16" : "f32";
    // i8a8 変種（設計 §7 の波 1 + 波 2）。**② 行統計は f32 のまま**（S の格納形も f32 の
    // ままなので、②③ が読む S は ① がどちらの経路で書いても同型）。
    // MUST: 適格判定は**段ごとに独立**。① は q / k のパック方向が D・③ は P̃ / Vᵀ の
    // パック方向が N なので、条件も別々になる（両方満たさない形・片方だけの形が実在する）。
    const i8a8 = this.#state.attentionCompute === "i8a8";
    const qkI8a8 = i8a8 && depth % 4 === 0;
    const pvI8a8 = i8a8 && cols % 4 === 0;
    // S の**格納形**（案 γ 波 1 — 計算形と直交する第 2 の軸）。`pack2x16float` の 2 要素／語
    // なので、書き手 ①QK が v4 経路を取る形（`D % 4 == 0 && N % 4 == 0`）だけが適格で、
    // 非適格は f32 格納へ**沈黙で**縮退する（検出器はパイプラインキー）。
    // MUST: 3 カーネルが**同時に**切り替わる — S の格納形が書き手と読み手で一致していなければ
    // ならないので、段ごとの混成はあり得ない（i8a8 の適格判定が段ごとに独立なのとは別の話で、
    // ①が i8a8 でも f32 でも S の格納形は 1 つに決まる）。
    const scoreStorage: ScoreStorage =
      this.#state.attentionScoreStorage === "f16" && attentionScoreUsesF16(depth, cols)
        ? "f16"
        : "f32";

    // 加算 mask（省略可能な第 4 入力 — 契約は src/ops.ts）。① の epilogue だけの軸で、
    // ②③ からは「S が既に mask 済み」に見える。
    const mask = binds[3];
    // MUST: i8a8 の ①QK は別カーネル（src/kernels/attention-i8a8.ts）で epilogue を持たない。
    // 黙って f32 経路へ落とすと「i8a8 を頼んだのに効かない」沈黙になり、mask を落とすと
    // 値が壊れるので、組み合わせそのものを**一時バッファを取る前に** fail loudly にする。
    if (mask !== undefined && qkI8a8) {
      throw new ExecutionError(
        `${where}: 加算 mask 付きの attention は attentionCompute 'i8a8' と組めない` +
          "（①QK の i8a8 変種は mask の epilogue を持たない — attentionCompute を 'f32' か " +
          "'f16' にすること）",
      );
    }

    // S[batch, M, N] と行統計 [batch·M, 2]。O は実行相が確保済みなので、峰は
    // O + S + 統計（分解経路の S + P + 恒等 expand の 3 枚から 1 枚ぶん減る）。
    // f16 変種（`:c16` の array<f16> / s16 の pack2x16float）では S が半分のバイト数になる
    // （1024px の DiT で 1,073.7MB → 536.9MB）。
    const scores = builder.allocTemp(
      batch * rows * cols * (compute === "f16" ? 2 : scoreStorageBytes(scoreStorage)),
    );
    const stats = builder.allocTemp(batch * rows * ATTENTION_STATS_STRIDE * 4);

    // ① QK gemm — 縮約次元は D、出力の列は N。
    if (qkI8a8) {
      await this.#buildAttentionQkI8a8(
        builder,
        binds,
        scores,
        scoreStorage,
        { batch, rows, cols, depth },
        scale,
        where,
      );
    } else {
      const qkV4 = gemmUsesVec4(depth, cols);
      const hasMask = mask !== undefined;
      const qkKey = attentionQkKey(qkV4, compute, scoreStorage, hasMask);
      const { pipeline: qkPipeline, layout: qkLayout } = await this.#state.cache.get(
        qkKey,
        attentionQkWgsl(qkV4, compute, scoreStorage, hasMask),
      );
      const geometry = defaultGemmGeometry();
      builder.dispatch({
        key: qkKey,
        pipeline: qkPipeline,
        layout: qkLayout,
        params: this.#writeParams(
          attentionQkParams(rows, cols, depth, scale),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: binds[0] },
          { binding: 2, source: binds[1] },
          { binding: 3, source: scores },
          ...(mask === undefined ? [] : [{ binding: ATTENTION_QK_MASK_BINDING, source: mask }]),
        ],
        workgroups: [
          tiledWorkgroups(cols, gemmTileN(geometry), limit, `${where} ①QK`),
          tiledWorkgroups(rows, gemmTileM(geometry), limit, `${where} ①QK`),
          tiledWorkgroups(batch, 1, limit, `${where} ①QK`),
        ],
      });
    }

    // ② 行統計 — 1 行 = 1 workgroup で、行方向は grid-stride（softmax と同じ形）。
    // S を 1 回だけ読む regcache 変種は dim 依存の生成なので、`epc` がキー軸に増える
    // （値はどちらもビット同一 — src/kernels/attention.ts）。
    const statsRegCache = attentionStatsRegCache(cols);
    const statsKey = attentionStatsKey(compute, scoreStorage, statsRegCache);
    const { pipeline: statsPipeline, layout: statsLayout } = await this.#state.cache.get(
      statsKey,
      attentionStatsWgsl(compute, scoreStorage, statsRegCache),
    );
    builder.dispatch({
      key: statsKey,
      pipeline: statsPipeline,
      layout: statsLayout,
      params: this.#writeParams(
        attentionStatsParams(batch * rows, cols),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [{ binding: 1, source: scores }, { binding: 2, source: stats }],
      workgroups: [gridStrideWorkgroups(batch * rows, 1, limit), 1, 1],
    });

    // ③ PV gemm — 縮約次元は N、出力の列は D。
    if (pvI8a8) {
      await this.#buildAttentionPvI8a8(
        builder,
        binds,
        scores,
        scoreStorage,
        stats,
        out,
        { batch, rows, cols, depth },
        where,
      );
    } else {
      const pvV4 = gemmUsesVec4(cols, depth);
      const pvKey = attentionPvKey(pvV4, compute, scoreStorage);
      const { pipeline: pvPipeline, layout: pvLayout } = await this.#state.cache.get(
        pvKey,
        attentionPvWgsl(pvV4, compute, scoreStorage),
      );
      const geometry = defaultGemmGeometry();
      builder.dispatch({
        key: pvKey,
        pipeline: pvPipeline,
        layout: pvLayout,
        params: this.#writeParams(attentionPvParams(rows, depth, cols), PARAMS_UNIFORM_USAGE),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: binds[2] },
          { binding: 3, source: stats },
          { binding: 4, source: out },
        ],
        workgroups: [
          tiledWorkgroups(depth, gemmTileN(geometry), limit, `${where} ③PV`),
          tiledWorkgroups(rows, gemmTileM(geometry), limit, `${where} ③PV`),
          tiledWorkgroups(batch, 1, limit, `${where} ③PV`),
        ],
      });
    }

    // MUST: ノード境界で一時バッファを返す（アリーナの不変条件 — 抜けると assertDrained で
    // 落ちるか、プール再利用から外れて peak が過大に出る）。
    builder.releaseTemp(stats);
    builder.releaseTemp(scores);
  }

  /**
   * 融合 attention ①QK の **i8a8 変種**（opt-in — {@link SessionOptions.attentionCompute}）。
   * ① が 1 dispatch から **3 dispatch** に増える（ノード全体では 3 → 5）:
   *
   * (a) `quantize_rows`（q を per-token i8 へ）→ (b) `quantize_rows`（k を per-token i8 へ）→
   * (c) i8a8 GEMM（整数内積 + dequant）。
   *
   * 量子化カーネルは linear の w8a8 と**同じ 1 本**（`QUANTIZE_ROWS_KEY` を共有する — 縮約軸
   * D が q / k とも最内連続なので、行 = token の per-token 量子化がそのまま要求どおりの形に
   * なる）。診断では linear の活性量子化と合算されるので、内訳は E2E のキー本数検査で担保する。
   *
   * MUST: 一時バッファ（`qq` / `qs` / `kq` / `ks`）は宣言 → 末尾で解放する。
   * MUST: `D` のオーバフロー門は fail loudly（黙って通すと i32 の巻き戻りで符号ごと化ける。
   * 実測形の D ≤ 384 に対して門は桁で余裕があるが、置かないと退行の受け皿が消える）。
   */
  async #buildAttentionQkI8a8(
    builder: StepRecipeBuilder,
    binds: readonly BindingSource[],
    scores: TempSource,
    scoreStorage: ScoreStorage,
    shape: {
      readonly batch: number;
      readonly rows: number;
      readonly cols: number;
      readonly depth: number;
    },
    scale: number,
    where: string,
  ): Promise<void> {
    const { batch, rows, cols, depth } = shape;
    if (depth > LINEAR_I8A8_MAX_K) {
      throw new ExecutionError(
        `${where}: D=${depth} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
          "（attentionCompute を 'f32' にすること）",
      );
    }
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // 量子化した q / k（i8 を 4 詰め）と per-token scale。ノード内で閉じた一時領域で、
    // q 側は**行** scale・k 側は**出力列**の scale になる（同じ per-token 量子化の別の読み方）。
    const qq = builder.allocTemp(Math.max(4, batch * rows * depth));
    const qs = builder.allocTemp(Math.max(4, batch * rows * 4));
    const kq = builder.allocTemp(Math.max(4, batch * cols * depth));
    const ks = builder.allocTemp(Math.max(4, batch * cols * 4));

    // (a)(b) 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await this.#state.cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    const quantize = (
      source: BindingSource,
      payload: TempSource,
      scales: TempSource,
      count: number,
    ): void => {
      builder.dispatch({
        key: QUANTIZE_ROWS_KEY,
        pipeline: quantizePipeline,
        layout: quantizeLayout,
        params: this.#writeParams(quantizeRowsParams(count, depth), PARAMS_UNIFORM_USAGE),
        bindings: [
          { binding: 1, source },
          { binding: 2, source: payload },
          { binding: 3, source: scales },
        ],
        workgroups: [gridStrideWorkgroups(count, 1, limit), 1, 1],
      });
    };
    quantize(binds[0], qq, qs, batch * rows);
    quantize(binds[1], kq, ks, batch * cols);

    // (c) 整数内積の GEMM（半スケールは dequant 側で q / k の両方へ — 設計 §2.1）。
    // 幾何は ③PV と**別に**選ぶ（③ だけ N = D の 1 タイル化が勝つ — 実測）。
    const v4 = attentionQkI8a8UsesVec4(cols);
    const dp4a = this.#state.i8a8Dot === "dp4a";
    const geometry = defaultI8a8Geometry("attention_qk");
    const key = attentionQkI8a8Key(v4, dp4a, scoreStorage, geometry);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      attentionQkI8a8Wgsl(v4, dp4a, scoreStorage, geometry),
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params: this.#writeParams(
        attentionQkI8a8Params(rows, cols, depth, scale),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source: qq },
        { binding: 2, source: kq },
        { binding: 3, source: scores },
        { binding: ATTENTION_QK_Q_SCALE_BINDING, source: qs },
        { binding: ATTENTION_QK_K_SCALE_BINDING, source: ks },
      ],
      workgroups: [
        tiledWorkgroups(cols, i8a8TileN(geometry), limit, `${where} ①QK i8a8`),
        tiledWorkgroups(rows, i8a8TileM(geometry), limit, `${where} ①QK i8a8`),
        tiledWorkgroups(batch, 1, limit, `${where} ①QK i8a8`),
      ],
    });

    // MUST: ノード境界で一時バッファを返す（確保と破棄を 1 箇所へ — ADR 0004）。
    builder.releaseTemp(ks);
    builder.releaseTemp(kq);
    builder.releaseTemp(qs);
    builder.releaseTemp(qq);
  }

  /**
   * 融合 attention ③PV の **i8a8 変種**（opt-in — {@link SessionOptions.attentionCompute}）。
   * ③ が 1 dispatch から **3 dispatch** に増える（①も i8a8 ならノード全体で 3 → 7）:
   *
   * (a) `strided`（v`[B·H,N,D]` → Vᵀ`[B·H,D,N]` の permute）→ (b) `quantize_rows`
   * （Vᵀ を行 = `(b,h,d)` で量子化）→ (c) i8a8 GEMM（整数内積 + dequant）。
   *
   * **新カーネルを 1 本も作らずに per-column 量子化が得られる**のがこの並びの要点（設計 §2.3）:
   * Vᵀ の「行」は `(b,h,d)` なので `quantize_rows` の per-token 量子化がそのまま
   * **V の per-column（N 全体の amax）scale** になり、同時に dp4a が要求する **N 連続パック**も
   * 手に入る。MUST: V を転置せずに量子化してはならない — scale が縮約軸 n の上で変わり、
   * `f32(acc)·s` 形の前提（s が n に依存しない）が壊れる（例外の出ない誤値）。
   *
   * P̃ 側は量子化カーネルを通らない（A タイル充填が `round(127·exp(S−m))` を作る）ので、
   * dispatch も一時バッファも増えない — ②行統計は f32 のまま 1 バイトも変えない。
   *
   * MUST: 一時バッファ（`vt` / `vq` / `vs`）は宣言 → 末尾で解放する。
   * MUST: `N` のオーバフロー門は fail loudly（|acc| ≤ N·127²。実測形の最大 N = 16,384 に対し
   * 門は桁で余裕があるが、置かないと退行の受け皿が消える）。
   */
  async #buildAttentionPvI8a8(
    builder: StepRecipeBuilder,
    binds: readonly BindingSource[],
    scores: TempSource,
    scoreStorage: ScoreStorage,
    stats: TempSource,
    out: BindingSource,
    shape: {
      readonly batch: number;
      readonly rows: number;
      readonly cols: number;
      readonly depth: number;
    },
    where: string,
  ): Promise<void> {
    const { batch, rows, cols, depth } = shape;
    if (cols > LINEAR_I8A8_MAX_K) {
      throw new ExecutionError(
        `${where}: N=${cols} が i8a8 経路の i32 縮約の門 ${LINEAR_I8A8_MAX_K} を超える` +
          "（attentionCompute を 'f32' にすること）",
      );
    }
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // Vᵀ（f32・permute の実体化）と、その量子化結果。どれもノード内で閉じた一時領域。
    const vt = builder.allocTemp(Math.max(4, batch * depth * cols * 4));
    const vq = builder.allocTemp(Math.max(4, batch * depth * cols));
    const vs = builder.allocTemp(Math.max(4, batch * depth * 4));

    // (a) v[B·H,N,D] → Vᵀ[B·H,D,N]（既存の strided 読みコピー族 — permute そのもの）。
    // MUST: stride は**入力** `[B·H,N,D]` の連続 stride から組む（出力 shape から組むと
    // D == N のときだけ一致する。実測形は D != N なので露見するが、単体テストが本来の検出器）。
    const stridedSpec = { dtype: "f32" } as const;
    const permuteKey = stridedKey(stridedSpec);
    const { pipeline: permutePipeline, layout: permuteLayout } = await this.#state.cache.get(
      permuteKey,
      stridedWgsl(stridedSpec),
    );
    builder.dispatch({
      key: permuteKey,
      pipeline: permutePipeline,
      layout: permuteLayout,
      params: this.#writeParams(
        stridedParams(
          [batch, depth, cols],
          permuteSrcStrides([batch, cols, depth], [0, 2, 1]),
          0,
        ),
        PARAMS_STORAGE_USAGE,
      ),
      bindings: [{ binding: 1, source: binds[2] }, { binding: 2, source: vt }],
      workgroups: [
        gridStrideWorkgroups(batch * depth * cols, STRIDED_WORKGROUP_SIZE, limit),
        1,
        1,
      ],
    });

    // (b) Vᵀ の量子化（行 = (b,h,d)・行長 N — per-column scale と N 連続パックが同時に出る）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await this.#state.cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    builder.dispatch({
      key: QUANTIZE_ROWS_KEY,
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      params: this.#writeParams(quantizeRowsParams(batch * depth, cols), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: vt },
        { binding: 2, source: vq },
        { binding: 3, source: vs },
      ],
      workgroups: [gridStrideWorkgroups(batch * depth, 1, limit), 1, 1],
    });

    // (c) 整数内積の GEMM（P̃ は A タイル充填で作る = 非実体化のまま）
    const v4 = attentionPvI8a8UsesVec4(depth);
    const dp4a = this.#state.i8a8Dot === "dp4a";
    const geometry = defaultI8a8Geometry("attention_pv");
    const key = attentionPvI8a8Key(v4, dp4a, scoreStorage, geometry);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      attentionPvI8a8Wgsl(v4, dp4a, scoreStorage, geometry),
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params: this.#writeParams(attentionPvI8a8Params(rows, depth, cols), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: vq },
        { binding: 3, source: stats },
        { binding: 4, source: out },
        { binding: ATTENTION_PV_V_SCALE_BINDING, source: vs },
      ],
      workgroups: [
        tiledWorkgroups(depth, i8a8TileN(geometry), limit, `${where} ③PV i8a8`),
        tiledWorkgroups(rows, i8a8TileM(geometry), limit, `${where} ③PV i8a8`),
        tiledWorkgroups(batch, 1, limit, `${where} ③PV i8a8`),
      ],
    });

    // MUST: ノード境界で一時バッファを返す（確保と破棄を 1 箇所へ — ADR 0004）。
    builder.releaseTemp(vs);
    builder.releaseTemp(vq);
    builder.releaseTemp(vt);
  }

  /**
   * embedding（行 gather）。範囲外添字の扱いは src/kernels/embedding.ts の裁定
   * （GPU は NaN 汚染 / CPU 参照は throw）。attrs の padding_idx は forward に効かないので
   * カーネルへ渡さない。
   */
  async #buildEmbedding(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const weight = step.inputShapes[0];
    const count = numel(step.outputShape);
    const weightStorage = this.#weightStorage(step);
    const key = embeddingKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(key, embeddingWgsl(weightStorage));
    const params = this.#writeParams(
      embeddingParams(count, weight[1], weight[0]),
      PARAMS_UNIFORM_USAGE,
    );
    const groups = gridStrideWorkgroups(
      count,
      EMBEDDING_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
        ...this.#weightScaleBindings(step, weightStorage, EMBEDDING_SCALE_BINDING),
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /**
   * masked_fill。出力と x は同形・連続で、mask だけを右詰め broadcast の stride で読む。
   * stride の組み立ては strided 族の expand と同じ規則（{@link expandSrcStrides}）。
   */
  async #buildMaskedFill(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const outShape = step.outputShape;
    // 規則は expand と同一だが、診断の主語は masked_fill の側に付け替える（グラフに expand が
    // 無いのに「expand の入力」と出ると原因の当たりを外す）。
    const maskStrides = expandSrcStrides(step.inputShapes[1], outShape, {
      src: "masked_fill の mask",
      out: "masked_fill の出力",
    });
    const value = maskedFillValue(step.node.attrs, `nodes (${step.node.op})`);
    const { pipeline, layout } = await this.#state.cache.get(
      MASKED_FILL_KEY,
      MASKED_FILL_WGSL,
    );
    const params = this.#writeParams(
      maskedFillParams(outShape, maskStrides, value),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(outShape),
      MASKED_FILL_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key: MASKED_FILL_KEY,
      pipeline,
      layout,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: out },
      ],
      workgroups: [groups, 1, 1],
    });
  }

  /** conv1d（直接畳み込み、groups / dilation は attrs）。1 スレッド = 1 出力要素の grid-stride。 */
  async #buildConv1d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding, dilation, groups } = conv1dAttrs(
      step.node.attrs,
      `nodes (${step.node.op})`,
    );
    const weightStorage = this.#weightStorage(step);
    const key = conv1dKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(key, conv1dWgsl(weightStorage));
    const params = this.#writeParams(
      conv1dParams({
        batch: outShape[0],
        channelsIn: x[1],
        channelsOut: outShape[1],
        lengthIn: x[2],
        lengthOut: outShape[2],
        kernel: weight[2],
        stride,
        padding,
        dilation,
        groups,
      }),
      PARAMS_UNIFORM_USAGE,
    );
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV1D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV1D_SCALE_BINDING),
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * conv2d（stride / padding / dilation は H/W の 2 成分）。**groups で 2 カーネルを踏み分ける**
   * （ADR 0024）: `groups == 1` は implicit GEMM、`groups > 1` は直接畳み込み。
   *
   * MUST: Kh / Kw は**重みの第 3 / 第 4 軸**をこの順で読む。入れ替えても正方カーネルでは
   * 数値が一致するので、テストは Kh ≠ Kw の形で固定する（src/kernels/conv2d.ts）。
   */
  async #buildConv2d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding, dilation, groups } = conv2dAttrs(
      step.node.attrs,
      `nodes (${step.node.op})`,
    );
    const dims: Conv2dDims = {
      batch: outShape[0],
      channelsIn: x[1],
      channelsOut: outShape[1],
      heightIn: x[2],
      widthIn: x[3],
      heightOut: outShape[2],
      widthOut: outShape[3],
      kernelH: weight[2],
      kernelW: weight[3],
      strideH: stride[0],
      strideW: stride[1],
      paddingH: padding[0],
      paddingW: padding[1],
      dilationH: dilation[0],
      dilationW: dilation[1],
      groups,
    };
    const weightStorage = this.#weightStorage(step);
    if (groups === 1) {
      await this.#buildConv2dIgemm(step, binds, out, builder, dims, weightStorage);
      return;
    }
    const key = conv2dKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(key, conv2dWgsl(weightStorage));
    const params = this.#writeParams(conv2dParams(dims), PARAMS_UNIFORM_USAGE);
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV2D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV2D_SCALE_BINDING),
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * conv2d の implicit GEMM（`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — ADR 0024）。
   *
   * MUST: GEMM 骨格と同じ「1 workgroup = 1 出力タイル」なので、dispatch 上限超過は
   * fail loudly（grid-stride で縮退させるとタイルが欠落し、full-write が黙って壊れる）。
   * MUST: バッチは **z 軸**（bmm と同じ）。N 側へ畳むと出力が `[Cout][B·Hout·Wout]` になり
   * NCHW と軸が入れ替わる — B = 1 でだけ一致するので実測形では露見しない。
   *
   * m タイルは形状の関数（{@link conv2dIgemmMTile}）で 64 行 / 32 行を選ぶ（ADR 0024 隣接）。
   * **n タイルは常に 64**（2048px の n 上限超過の扱いを動かさない）。どちらのタイル形でも
   * 出力はビット同一なので、これは純粋な dispatch の割り直しで数値契約に触れない。
   */
  async #buildConv2dIgemm(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
    dims: Conv2dDims,
    weightStorage: WeightStorage,
  ): Promise<void> {
    const m = dims.channelsOut;
    const n = dims.heightOut * dims.widthOut;
    const kFlat = dims.channelsIn * dims.kernelH * dims.kernelW;
    const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
    const mTile = conv2dIgemmMTile(m);
    // 生成・キーと同じ解決点から幾何を引く（dispatch のタイル辺が WGSL の辺と構造的に一致）。
    const geometry = gemmMTileGeometry(mTile);
    const key = conv2dIgemmKey(weightStorage, v4, mTile);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      conv2dIgemmWgsl(weightStorage, v4, mTile),
    );
    const params = this.#writeParams(conv2dIgemmParams(dims), PARAMS_UNIFORM_USAGE);
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `conv2d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV2D_SCALE_BINDING),
      ],
      workgroups: [
        tiledWorkgroups(n, gemmTileN(geometry), limit, where),
        tiledWorkgroups(m, gemmTileM(geometry), limit, where),
        tiledWorkgroups(dims.batch, 1, limit, where),
      ],
    });
  }

  /**
   * conv_transpose1d（gather 形）。1 スレッド = 1 出力要素で出力全域を書く（ADR 0014）。
   *
   * MUST: Cin は**重みの第 1 軸**（`[Cin, Cout, K]`）。x[1] と一致することは契約検査済みだが、
   * ここで weight[0] を使うのは「転置レイアウトの正本は重み側」という読みを崩さないため。
   */
  async #buildConvTranspose1d(
    step: NodePlan,
    binds: readonly BindingSource[],
    out: BindingSource,
    builder: StepRecipeBuilder,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding } = convTranspose1dAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const weightStorage = this.#weightStorage(step);
    const key = convTranspose1dKey(weightStorage);
    const { pipeline, layout } = await this.#state.cache.get(
      key,
      convTranspose1dWgsl(weightStorage),
    );
    const params = this.#writeParams(
      convTranspose1dParams({
        batch: outShape[0],
        channelsIn: weight[0],
        channelsOut: outShape[1],
        lengthIn: x[2],
        lengthOut: outShape[2],
        kernel: weight[2],
        stride,
        padding,
      }),
      PARAMS_UNIFORM_USAGE,
    );
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV_TRANSPOSE1D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      params,
      bindings: [
        ...binds.map((source, index) => ({ binding: index + 1, source })),
        { binding: 4, source: out },
        ...this.#weightScaleBindings(step, weightStorage, CONV_TRANSPOSE1D_SCALE_BINDING),
      ],
      workgroups: [workgroups, 1, 1],
    });
  }

  /**
   * dispatch の params uniform / storage を確保して書く。**内容そのものをキーにした Session
   * 常駐キャッシュ**を通すので、同じ内容の params は Session の生涯で 1 度しか確保・転送
   * されない（params の全バイトは グラフ・node.attrs・解決済み shape の純関数で、実行時の
   * テンソル値に依存しない）。キーは usage と全要素を連結した文字列そのもの — ハッシュでは
   * ないので衝突が原理的に無い。
   *
   * MUST: 確保先は **weights アリーナ（Session 常駐）**で、RunArena からは取らない。run ごとの
   * アリーナは run 末尾で破棄されるため、そこから配るとキャッシュが破棄済みバッファを指す。
   * MUST: キャッシュしたバッファは**一度書いたら二度と書き換えない**。ADR 0004 が
   * `allocHostWritten` をプール対象外にしているのは「`queue.writeBuffer` が未 submit の先行
   * エンコードを追い越す」ためだが、ここは同じバッファへの 2 度目の writeBuffer が存在しない
   * ので追い越しハザードが原理的に生じない（プール再利用とは別レイヤの、内容同一性による
   * 共有）。
   * MUST: 破棄は weights アリーナの dispose に相乗りする（`Session.dispose` →
   * `RunArena.destroy` の flush-before-destroy）。キャッシュ専用の破棄経路を新設すると
   * flush の担い手が 2 つになり、どちらが先に走るかで破棄済みバッファ参照の submit が生まれる。
   * NOTE: 失敗 run（`scheduler.discard` 経路）の後もキャッシュは有効。バッファの内容は不変で、
   * discard が捨てるのは未 submit のエンコードだけなので、params の値は影響を受けない。
   */
  #writeParams(params: Uint32Array<ArrayBuffer>, usage: number): GPUBuffer {
    const key = `u${usage}:${params.join(",")}`;
    const cached = this.#state.paramsCache.get(key);
    if (cached !== undefined) {
      this.#paramsReuseCount += 1;
      return cached;
    }
    const buffer = this.#state.weights.allocHostWritten(params.byteLength, usage);
    this.#state.gpu.device.queue.writeBuffer(buffer, 0, params);
    this.#state.paramsCache.set(key, buffer);
    this.#paramsAllocCount += 1;
    return buffer;
  }

  /**
   * 読み戻してよい実体か。
   *
   * MUST: slot backing の実体は **pin された slot だけ**（グラフ出力として確保され、プールへ
   * 返らない slot）。他の slot は run 中に別の値へ配り直されるので、非 backed run で
   * `RunArena.isReadable` が中間値を拒むのと同じ規律をここで保つ — backing のバッファは
   * アリーナの所有外なので、放置すると `isReadable` が「プール対象外」として素通しにする。
   */
  #isReadable(buffer: GPUBuffer, arena: RunArena, backing: ActiveBacking | undefined): boolean {
    if (backing !== undefined && backing.owned.has(buffer)) return backing.readable.has(buffer);
    return arena.isReadable(buffer);
  }

  /** MUST: 読み戻すのはグラフ出力のみ。中間値はプール再利用で内容が入れ替わっている。 */
  async #readOutputs(
    env: ReadonlyMap<string, GPUBuffer>,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    backing: ActiveBacking | undefined,
  ): Promise<RunOutputs> {
    const device = this.#state.gpu.device;
    const staged: {
      readonly name: string;
      readonly shape: readonly number[];
      readonly count: number;
      readonly staging: GPUBuffer;
    }[] = [];
    // MUST: この copy → submit も errorScope に入れる（run 本体のスコープは pop 済みで、
    // ここは独自 encoder の別 submit になる）。COPY_SRC 欠落等の validation 失敗は例外に
    // ならず、読み戻しが全 0 のまま静かに返る。staging の確保が device の余力を超える形も
    // あるため out-of-memory と両建てにする。
    // MUST NOT: push から pop の発行までに await を挟まない。この区間を同期に保つことが、
    // device 単位ロックを取らずに LIFO の交錯を防いでいる根拠になっている。
    pushFailureScopes(device);
    let popped = false;
    try {
      const encoder = device.createCommandEncoder();
      for (const name of this.#state.model.graph.outputs) {
        // MUST: initializer も引く。IR は graph.outputs に initializer 名を書くことを許して
        // おり（format/ir.ts の定義済み検査）、その値は env（run 寿命）には載らない。
        const buffer = env.get(name) ?? this.#state.weightBuffers.get(name);
        if (buffer === undefined) throw new ExecutionError(`グラフ出力 '${name}' のバッファが無い`);
        if (!this.#isReadable(buffer, arena, backing)) {
          throw new ExecutionError(`グラフ出力 '${name}' がピン留めされておらず読み戻せない`);
        }
        const shape = resolvedShape(shapes, name);
        const count = numel(shape);
        // MUST: 0 要素でもアリーナの最小サイズクラス（4 バイト）に合わせる。copy サイズは
        // 両バッファの実サイズ以下でなければならず、出力側も同じ下限で確保されている。
        const size = Math.max(4, count * 4);
        const staging = arena.allocHostRead(size);
        staged.push({ name, shape, count, staging });
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
      }
      device.queue.submit([encoder.finish()]);

      // MUST: mapAsync より前に pop を発行する（mapAsync 待ちの間に別の操作を吸わないため）。
      // mapAsync 自身はどのスコープにも入らないが、#runOnce のロック区間内なので、その失敗が
      // 他人のスコープに帰属することはない（run の GPU 操作は全てロック区間内 — 上記不変条件）。
      const pending = popFailureScopes(device, "出力の readback");
      popped = true;
      const failure = await pending;
      if (failure !== undefined) throw failure;

      // MUST: device 消失時 mapAsync は解決しない。待ち続けるとハングになるため競わせる。
      await this.#state.gpu.raceDeviceLost(
        Promise.all(staged.map((item) => item.staging.mapAsync(GPUMapMode.READ))),
        "readback",
      );

      // MUST: グラフ出力名由来のキーを蓄積する器は null プロトタイプ（同上）。素の `{}` では
      // 出力名が "__proto__" のとき Tensor が [[Prototype]] に化け、その出力だけ own property
      // として現れないまま（Object.keys から消えたまま）返る沈黙欠落になる。
      const outputs: Record<string, Tensor> = Object.create(null);
      for (const item of staged) {
        const copy = item.staging.getMappedRange().slice(0);
        item.staging.unmap();
        // MUST: 読み戻す TypedArray は宣言 dtype から決める（要素は全型 4 バイトなので、
        // f32 固定にすると i32 / bool の出力が黙ってビット列の読み替えになる）。
        outputs[item.name] = hostTensor(
          this.#declaredDtype(item.name),
          item.shape,
          copy,
          item.count,
        );
      }
      return outputs;
    } finally {
      // `!popped` は本体が例外で抜けたときにだけ成立する。後始末でその例外を隠さない。
      // staging の破棄は RunArena が持つ（成功・失敗のどちらの経路でも #runOnce が
      // arena.destroy() まで必ず進む — ADR 0004「確保と破棄を 1 箇所へ」）。
      if (!popped) await discardFailureScopes(device);
    }
  }
}

/**
 * モデルを実行可能な Session にする。重みの GPU アップロードはこの async ステージで行う。
 */
export const createSession = (
  gpu: GpuContext,
  model: KarumeModel,
  options: SessionOptions = {},
): Promise<Session> => Session.create(gpu, model, options);
