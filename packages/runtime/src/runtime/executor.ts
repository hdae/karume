/**
 * 最小のグラフ実行器。
 *
 * 構造: 「計画（純関数・plan.ts）→ エンコード（GPU）→ submit → グラフ出力だけ readback」。
 * 出力 shape は 1 dispatch も出す前に全て確定し（静的形状 — ADR 0004）、想定外は全て
 * fail loudly にする（未対応 op / 契約外 dtype / broadcast 不可 / 未束縛シンボル / 宣言不一致）。
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
import { type ArenaStats, RunArena } from "../gpu/arena.ts";
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
import { GEMM_TILE, type GemmCompute, gemmUsesVec4 } from "../kernels/gemm.ts";
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

export type SessionDiagnostics = {
  readonly pipelineCount: number;
  readonly submit: SubmitStats;
  /** 重み（initializer）アリーナの実績。 */
  readonly weights: ArenaStats;
  /** 低精度格納の適格 / 適格外の内訳（ADR 0006 の常設診断）。 */
  readonly storage: StorageDiagnostics;
  /** 直近 run の中間バッファ実績。未実行なら undefined。 */
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

type SessionState = {
  readonly gpu: GpuContext;
  readonly model: KarumeModel;
  readonly cache: PipelineCache;
  readonly scheduler: SubmitScheduler;
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
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
    this.#disposal ??= this.#enqueue(() => this.#state.weights.destroy());
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
    const plan = planGraph(graph, bindSymbols(graph, inputShapes, bindings));
    // 融合の判定は GPU に触れない純関数（src/runtime/fusion.ts）。掴めなかったノードは素の
    // ままステップ列に並ぶので、この段は「速くなるか」だけを決め、正しさには関与しない。
    const fusion = planFusions(plan.nodes, {
      useCounts: this.#state.useCounts,
      outputNames: this.#state.outputNames,
    });
    this.#lastRunFusions = fusion.counts;

    const device = gpu.device;
    const arena = new RunArena(device, () => scheduler.flush());
    const env = new Map<string, GPUBuffer>(this.#state.weightBuffers);
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
      try {
        // 無効な bindGroup / dispatch は throw せず submit ごと失敗し、出力にプール残骸が
        // 残る沈黙故障になる。中間バッファの確保も同じ区間で out-of-memory を見る。
        pushFailureScopes(device);
        let popped = false;
        try {
          for (const spec of graph.inputs) {
            env.set(spec.name, this.#uploadInput(spec.name, inputs[spec.name], plan.shapes, arena));
          }
          for (const step of fusion.steps) await this.#encodeStep(step, env, arena);
          arena.assertDrained();

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

        outputs = await this.#readOutputs(env, plan.shapes, arena);
        this.#lastRun = arena.stats;
      } catch (cause) {
        // MUST: 後始末の失敗で本体の例外を上書きしない。原因は本体側にあり、destroy の
        // rejection（主因は device 消失）に差し替わると調査の起点が消える。
        await arena.destroy().catch(() => undefined);
        throw cause;
      }
      // 本体が通ったときは後始末の失敗をそのまま伝える（flush 未完了のまま返さない）。
      await arena.destroy();
      return outputs;
    });
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
  #weightScaleEntries(
    step: NodePlan,
    storage: WeightStorage,
    binding: number,
  ): readonly GPUBindGroupEntry[] {
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
    return [{ binding, resource: { buffer } }];
  }

  /** 値の宣言 dtype。引けないのはランタイム内部の不変条件破れ（declaredDtypes は全値を持つ）。 */
  #declaredDtype(name: string): IrDtype {
    const dtype = this.#state.dtypes.get(name);
    if (dtype === undefined) {
      throw new ExecutionError(`値 '${name}' の宣言 dtype が無い`);
    }
    return dtype;
  }

  #uploadInput(
    name: string,
    tensor: Tensor | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
  ): GPUBuffer {
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
    const buffer = arena.allocHostWritten(Math.max(4, count * 4), HOST_WRITTEN_USAGE);
    if (count > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, tensor.data);
    return buffer;
  }

  /**
   * 実行ステップ 1 つ（素のノード または 融合ステップ）のエンコード。
   *
   * MUST: 確保 → retain → 本体 → 入力の release（延べ）→ 定義ぶんの release、という簿記は
   * 両者で**この 1 本**に閉じる。融合ごとに手書きの解放簿記を置くと、アリーナの参照計数が
   * 融合の本数だけ別実装になり、1 本でもずれると例外なしの沈黙誤値になる（早すぎる解放なら
   * プール再利用で値が化け、多すぎれば peak が落ちない）。
   */
  async #encodeStep(
    step: ExecStep,
    env: Map<string, GPUBuffer>,
    arena: RunArena,
  ): Promise<void> {
    const outputName = step.kind === "node" ? step.plan.outputName : step.outputName;
    const outputShape = step.kind === "node" ? step.plan.outputShape : step.outputShape;
    // bind 面のオペランド順（重複無し）と解放簿記の延べ列は別物。素のノードでは
    // どちらも node.ins に一致し、融合ステップだけが 2 つを別々に宣言する。
    const bindNames = step.kind === "node" ? step.plan.node.ins : step.binds;
    const consumedNames = step.kind === "node" ? step.plan.node.ins : step.ins;
    const inputs = bindNames.map((name) => {
      const buffer = env.get(name);
      if (buffer === undefined) throw new ExecutionError(`値 '${name}' のバッファが無い`);
      return buffer;
    });
    // MUST: 出力の確保は当該 dispatch のエンコードより前（アリーナの不変条件）。この順序が
    // 「まだ読まれる入力が出力として配り直される」事故を構造的に防いでいる。
    //
    // reshape と恒等 expand は要素順を変えないので**入力バッファをそのまま出力の実体にする**
    // （別名 — ADR 0011）。dispatch も確保も出さない。要素数一致は planGraph が済ませているので、
    // 別名先の実バッファは宣言 shape ぶんの大きさを必ず満たす。
    const out = step.kind === "node" && step.aliasesInput
      ? inputs[0]
      : arena.allocStorage(numel(outputShape) * 4);
    // MUST: 別名でも retain は「定義ぶんの 1 + 出力値の消費回数」を**実バッファに積む**。
    // これで実バッファの参照数が「入力側の残り消費 + 出力側の消費」の和になり、別名越しの
    // 消費まで正確に数えられる。抜くと最終消費より早くプールへ戻り、次の確保に配り直された
    // 実体を後続が読む沈黙誤値になる（過剰に積めばプール再利用から外れて peak が落ちない）。
    arena.retain(out, this.#state.useCounts.get(outputName) ?? 0, {
      pinned: this.#state.outputNames.has(outputName),
    });
    env.set(outputName, out);

    if (step.kind === "node") {
      await this.#encodeNode(step.plan, step.aliasesInput, inputs, out, arena);
    } else {
      await this.#encodeFused(step, inputs, out, arena);
    }

    // MUST: 解放はステップ境界（当該ステップの全 dispatch をエンコードし終えた後）のみ。
    for (const name of consumedNames) {
      const buffer = env.get(name);
      if (buffer !== undefined) arena.release(buffer);
    }
    // MUST: retain が積んだ定義ぶんの 1 をここで返す。消費者ゼロの中間出力（グラフ出力にも
    // ならない到達不能な値）が解放されるのはこの 1 本だけで、抜けるとプール再利用から外れて
    // peakTransientBytes が実際より大きく出る。
    arena.release(out);
  }

  /**
   * 融合ステップの 1 dispatch。bind 面は「params, 入力…, 出力」で全ルール共通、params は
   * 16 バイトの uniform で固定（src/runtime/fusion.ts の {@link FusedDispatch}）。
   */
  async #encodeFused(
    step: FusedStep,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const { key, gridItems, workgroupSize } = step.dispatch;
    const pipeline = await this.#state.cache.get(key, step.dispatch.wgsl());
    const params = this.#writeParams(arena, step.dispatch.params, PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: inputs.length + 1, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      gridItems,
      workgroupSize,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
  }

  /** 素のノード 1 つの本体（確保・retain・解放は {@link #encodeStep} が済ませている）。 */
  async #encodeNode(
    step: NodePlan,
    aliasesInput: boolean,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    switch (step.contract.kind) {
      case "unary":
      case "binary":
        await this.#encodeElementwise(
          step,
          { op: step.contract.name, dtype: step.inputDtypes[0] },
          inputs,
          out,
          arena,
        );
        break;
      case "cast":
        await this.#encodeElementwise(
          step,
          { op: CAST_OP, dtype: step.inputDtypes[0], to: step.outputDtype },
          inputs,
          out,
          arena,
        );
        break;
      case "where":
        // 生成入力は**値スロット**の dtype（スロット 0 は条件で常に bool）。
        await this.#encodeElementwise(
          step,
          { op: WHERE_OP, dtype: step.inputDtypes[1] },
          inputs,
          out,
          arena,
        );
        break;
      case "cumsum":
        await this.#encodeCumsum(step, inputs, out, arena);
        break;
      case "matmul":
        await this.#encodeMatmul(step, inputs, out, arena);
        break;
      case "bmm":
        await this.#encodeBmm(step, inputs, out, arena);
        break;
      case "gather":
        await this.#encodeGather(step, inputs, out, arena);
        break;
      case "rowReduce":
        await this.#encodeRowReduce(step, step.contract.name, inputs, out, arena);
        break;
      case "permute":
      case "slice":
      case "symPrefixSlice":
        await this.#encodeStridedCopy(step, step.contract.kind, inputs, out, arena);
        break;
      case "expand":
        // 恒等 expand は別名化済み（0 dispatch）。複製軸が 1 つでもあれば実体化コピー。
        if (!aliasesInput) {
          await this.#encodeStridedCopy(step, step.contract.kind, inputs, out, arena);
        }
        break;
      case "cat":
        await this.#encodeCat(step, inputs, out, arena);
        break;
      case "pad":
        await this.#encodePad(step, inputs, out, arena);
        break;
      case "flip":
        await this.#encodeFlip(step, inputs, out, arena);
        break;
      case "linear":
        await this.#encodeLinear(step, inputs, out, arena);
        break;
      case "layerNorm":
        await this.#encodeLayerNorm(step, inputs, out, arena);
        break;
      case "rmsNorm":
        await this.#encodeRmsNorm(step, inputs, out, arena);
        break;
      case "softmax":
        await this.#encodeSoftmax(step, inputs, out, arena);
        break;
      case "attention":
        await this.#encodeAttention(step, inputs, out, arena);
        break;
      case "embedding":
        await this.#encodeEmbedding(step, inputs, out, arena);
        break;
      case "maskedFill":
        await this.#encodeMaskedFill(step, inputs, out, arena);
        break;
      case "conv1d":
        await this.#encodeConv1d(step, inputs, out, arena);
        break;
      case "conv2d":
        await this.#encodeConv2d(step, inputs, out, arena);
        break;
      case "convTranspose1d":
        await this.#encodeConvTranspose1d(step, inputs, out, arena);
        break;
      case "reshape":
        // 別名化は #encodeStep で済んでいる（この op は 1 dispatch も出さない）。
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

  async #encodeElementwise(
    step: NodePlan,
    element: ElementwiseOp,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    // rank 0（スカラ）は codegen の rank ≥ 1 契約に合わせて長さ 1 の 1 次元に正規化する。
    const outShape = step.outputShape.length === 0 ? [1] : [...step.outputShape];
    const rank = outShape.length;
    const spec: ElementwiseSpec = element.op === CAST_OP
      ? { op: CAST_OP, rank, dtype: element.dtype, to: element.to }
      : { op: element.op, rank, dtype: element.dtype };
    const key = elementwiseKey(spec);
    const pipeline = await this.#state.cache.get(key, elementwiseWgsl(spec));
    // attrs のスカラ（clamp の min/max など）は params の末尾に f32 で載る（並びは契約表）。
    const params = this.#writeParams(
      arena,
      elementwiseParams(
        outShape,
        step.inputShapes,
        scalarParamValues(step.contract, step.node.attrs, `nodes (${step.node.op})`),
      ),
      PARAMS_STORAGE_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: inputs.length + 1, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      numel(outShape),
      ELEMENTWISE_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
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
  async #encodeRowReduce(
    step: NodePlan,
    op: ReduceOpName,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const inputShape = step.inputShapes[0];
    const axis = reduceDim(step.node.attrs, `nodes (${step.node.op})`);
    // 要素型はキーに載る（bool の sum は u32 を読んで i32 の個数を書く — ADR 0009）。
    const spec = { op, dtype: step.inputDtypes[0] };
    const lastDim = axis === inputShape.length - 1;
    const outCount = numel(step.outputShape);
    const key = lastDim ? reduceKey(spec) : axisReduceKey(spec);
    const pipeline = await this.#state.cache.get(
      key,
      lastDim ? reduceWgsl(spec) : axisReduceWgsl(spec),
    );
    const inner = numel(inputShape.slice(axis + 1));
    const params = this.#writeParams(
      arena,
      lastDim
        ? reduceParams(outCount, inputShape[axis])
        : axisReduceParams(outCount, inputShape[axis], inner),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    // 行 reduce は 1 行 = 1 workgroup、軸 reduce は 1 スレッド = 1 出力。どちらも上限を
    // 超えたら縮退させ、カーネル側の grid-stride で回す。
    const groups = gridStrideWorkgroups(
      outCount,
      lastDim ? 1 : AXIS_REDUCE_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
  }

  /**
   * cumsum（最終次元の前縁和）。**1 invocation = 1 行**の逐次走査で、行方向を grid-stride で
   * 回す（形の根拠は src/kernels/cumsum.ts）。
   */
  async #encodeCumsum(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const pipeline = await this.#state.cache.get(CUMSUM_KEY, CUMSUM_WGSL);
    const params = this.#writeParams(arena, cumsumParams(rows, dim), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      rows,
      CUMSUM_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], CUMSUM_KEY);
  }

  /**
   * permute / 非恒等 expand / slice / sym_prefix_slice の実体化コピー（strided 読み 1 カーネル族 —
   * ADR 0011 / 0010 / 0014）。出力は常に連続で、入力側だけを stride で読む。expand の複製軸は
   * stride 0、sym_prefix_slice は **Tmax 形の入力**の連続 stride、slice は入力の連続 stride と
   * **開始位置 offset**。恒等 expand は別名化されるのでここへは来ない。
   */
  async #encodeStridedCopy(
    step: NodePlan,
    kind: "permute" | "expand" | "slice" | "symPrefixSlice",
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
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
    const pipeline = await this.#state.cache.get(key, stridedWgsl(spec));
    const params = this.#writeParams(
      arena,
      stridedParams(outShape, srcStrides, offset),
      PARAMS_STORAGE_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      numel(outShape),
      STRIDED_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
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
  async #encodeCat(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const outShape = step.outputShape;
    const where = `nodes (${step.node.op})`;
    const dim = catDim(step.node.attrs, where);
    const outStrides = catOutStrides(outShape);
    const spec = { dtype: step.outputDtype };
    const key = stridedWriteKey(spec);
    const pipeline = await this.#state.cache.get(key, stridedWriteWgsl(spec));
    let written = 0;
    for (const [index, buffer] of inputs.entries()) {
      const srcShape = step.inputShapes[index];
      const params = this.#writeParams(
        arena,
        stridedWriteParams(srcShape, outStrides, catOutOffset(outShape, dim, written)),
        PARAMS_STORAGE_USAGE,
      );
      const bindGroup = this.#state.gpu.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer } },
          { binding: 2, resource: { buffer: out } },
        ],
      });
      const groups = gridStrideWorkgroups(
        numel(srcShape),
        STRIDED_WRITE_WORKGROUP_SIZE,
        this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
      );
      this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
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
  async #encodePad(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const { left, right } = padAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const pipeline = await this.#state.cache.get(PAD_KEY, PAD_WGSL);
    const params = this.#writeParams(
      arena,
      padParams(numel(srcShape.slice(0, -1)), srcShape[srcShape.length - 1], left, right),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      numel(outShape),
      PAD_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], PAD_KEY);
  }

  /**
   * flip（静的軸の添字反転）。軸の位置は `[outer, len, inner]` の 3 分割に畳んで渡す
   * （rank を params に載せない — src/kernels/flip.ts）。
   */
  async #encodeFlip(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = flipDim(step.node.attrs, `nodes (${step.node.op})`);
    const pipeline = await this.#state.cache.get(FLIP_KEY, FLIP_WGSL);
    const params = this.#writeParams(
      arena,
      // MUST: 軸の前後で分ける（`slice(0, dim)` と `slice(dim + 1)`）。境界を 1 つずらすと
      // 反転する軸が隣にずれ、shape も要素数も変わらないまま値だけが誤る。
      flipParams(numel(shape.slice(0, dim)), shape[dim], numel(shape.slice(dim + 1))),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      numel(shape),
      FLIP_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], FLIP_KEY);
  }

  async #encodeMatmul(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const [a, b] = step.inputShapes;
    const [m, k] = a;
    const n = b[1];
    // v4（vec4 の読み書き）は形状から導く 1 ビット。encode 時に評価してキーと WGSL の
    // 両方へ渡す（同一キー ⇔ 同一バイト列は保たれる）。
    const v4 = gemmUsesVec4(k, n);
    const key = matmulKey(v4);
    const pipeline = await this.#state.cache.get(key, matmulWgsl(v4));
    const params = this.#writeParams(arena, matmulParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: inputs[1] } },
        { binding: 3, resource: { buffer: out } },
      ],
    });
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `matmul [${a.join(",")}] × [${b.join(",")}]`;
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(n, GEMM_TILE, limit, where),
      tiledWorkgroups(m, GEMM_TILE, limit, where),
      1,
    ], key);
  }

  /**
   * バッチ matmul（rank-3）。タイル 2 軸に加えて**バッチを z 軸**へ載せる。
   * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、3 軸とも上限超過は fail loudly。
   */
  async #encodeBmm(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const [a, b] = step.inputShapes;
    const [batch, m, k] = a;
    const n = b[2];
    const v4 = gemmUsesVec4(k, n);
    const key = bmmKey(v4);
    const pipeline = await this.#state.cache.get(key, bmmWgsl(v4));
    const params = this.#writeParams(arena, bmmParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: inputs[1] } },
        { binding: 3, resource: { buffer: out } },
      ],
    });
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `bmm [${a.join(",")}] × [${b.join(",")}]`;
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(n, GEMM_TILE, limit, where),
      tiledWorkgroups(m, GEMM_TILE, limit, where),
      // バッチは 1 workgroup = 1 バッチ。ここも縮退させるとバッチが丸ごと未書き込みになる。
      tiledWorkgroups(batch, 1, limit, where),
    ], key);
  }

  /**
   * 最終次元の gather。出力は連続で、`row = i / J` から `src[row * D + index[i]]` を引く。
   * 範囲外添字の扱いは src/kernels/gather.ts の裁定（GPU は NaN 汚染 / CPU 参照は throw）。
   */
  async #encodeGather(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const srcShape = step.inputShapes[0];
    const outShape = step.outputShape;
    const count = numel(outShape);
    const pipeline = await this.#state.cache.get(GATHER_KEY, GATHER_WGSL);
    const params = this.#writeParams(
      arena,
      gatherParams(count, outShape[outShape.length - 1], srcShape[srcShape.length - 1]),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: inputs[1] } },
        { binding: 3, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      count,
      GATHER_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], GATHER_KEY);
  }

  /**
   * linear（融合 op — ADR 0012）。先行次元を平坦化して `[m,k] × [k,n] + bias` の 2 次元
   * GEMM に落とす。重みは `[n,k]` の転置レイアウトのままカーネルが読む（転置コピー無し）。
   * MUST: matmul と同じ「1 workgroup = 1 タイル」なので、上限超過は fail loudly。
   */
  async #encodeLinear(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const n = weight[0];
    const k = weight[1];
    const m = numel(x.slice(0, -1));
    const weightStorage = this.#weightStorage(step);
    // w8a8 は **opt-in × i8 常駐 × k % 4 == 0** の 3 条件が揃ったときだけ（ADR 0025 予定）。
    // 既定の "f32" では 1 バイトも挙動が変わらない。
    if (this.#state.linearCompute === "i8a8" && weightStorage === "i8" && k % 4 === 0) {
      await this.#encodeLinearI8a8(step, inputs, out, arena, m, n, k);
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
    const key = linearKey(weightStorage, v4, compute);
    const pipeline = await this.#state.cache.get(key, linearWgsl(weightStorage, v4, compute));
    const params = this.#writeParams(arena, linearParams(m, n, k), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 4, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, weightStorage, LINEAR_SCALE_BINDING),
      ],
    });
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `linear [${x.join(",")}] × [${weight.join(",")}]`;
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(n, GEMM_TILE, limit, where),
      tiledWorkgroups(m, GEMM_TILE, limit, where),
      1,
    ], key);
  }

  /**
   * linear の **w8a8 変種**（opt-in — {@link SessionOptions.linearCompute}）。**1 ノード =
   * 2 dispatch**（融合 attention と同じ「複数 dispatch で 1 ノード」の扱い）:
   *
   * ① `quantize_rows`（活性を per-token i8 へ・行方向 grid-stride）→ ② i8a8 GEMM（整数内積・
   * 1 workgroup = 1 出力タイルなので上限超過は fail loudly）。
   *
   * MUST: 一時バッファ（`xq` / `xs`）は `retain(…, 0)` → ノード末尾で `release` する。これで
   * アリーナの参照計数が閉じ（`assertDrained`）、失敗経路でも `arena.destroy()` が拾う。
   * MUST: `k` のオーバフロー門は fail loudly。黙って通すと i32 の巻き戻りで符号ごと化ける。
   */
  async #encodeLinearI8a8(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
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
    const device = this.#state.gpu.device;
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // 量子化した活性 `xq`（i8 を 4 詰め）と per-token scale `xs`。ノード内で閉じた一時領域。
    const xq = arena.allocStorage(Math.max(4, m * (k / 4) * 4));
    arena.retain(xq, 0);
    const xs = arena.allocStorage(Math.max(4, m * 4));
    arena.retain(xs, 0);

    // ① 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
    const quantizePipeline = await this.#state.cache.get(QUANTIZE_ROWS_KEY, QUANTIZE_ROWS_WGSL);
    const quantizeParams = this.#writeParams(
      arena,
      quantizeRowsParams(m, k),
      PARAMS_UNIFORM_USAGE,
    );
    const quantizeBindGroup = device.createBindGroup({
      layout: quantizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: quantizeParams } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: xq } },
        { binding: 3, resource: { buffer: xs } },
      ],
    });
    this.#state.scheduler.dispatch(quantizePipeline, quantizeBindGroup, [
      gridStrideWorkgroups(m, 1, limit),
      1,
      1,
    ], QUANTIZE_ROWS_KEY);

    // ② 整数内積の GEMM。タイル幾何は op → 幾何の純関数が決める（src/kernels/i8a8-geometry.ts）
    // — キーに載るので「同一キー → バイト同一 WGSL」は保たれる。
    const v4 = linearI8a8UsesVec4(n);
    const geometry = defaultI8a8Geometry("linear");
    const key = linearI8a8Key(v4, this.#state.i8a8Dot === "dp4a", geometry);
    const pipeline = await this.#state.cache.get(
      key,
      linearI8a8Wgsl(v4, this.#state.i8a8Dot === "dp4a", geometry),
    );
    const params = this.#writeParams(arena, linearI8a8Params(m, n, k), PARAMS_UNIFORM_USAGE);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: xq } },
        { binding: 2, resource: { buffer: inputs[1] } },
        { binding: 3, resource: { buffer: inputs[2] } },
        { binding: 4, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, "i8", LINEAR_SCALE_BINDING),
        { binding: LINEAR_ACT_SCALE_BINDING, resource: { buffer: xs } },
      ],
    });
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(n, i8a8TileN(geometry), limit, where),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, where),
      1,
    ], key);

    // MUST: ノード境界で一時バッファを返す（アリーナの不変条件）。
    arena.release(xs);
    arena.release(xq);
  }

  /** layer_norm（最終次元・affine あり）。1 行 = 1 workgroup で、行方向は grid-stride。 */
  async #encodeLayerNorm(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const { eps } = layerNormAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const pipeline = await this.#state.cache.get(LAYER_NORM_KEY, LAYER_NORM_WGSL);
    const params = this.#writeParams(
      arena,
      layerNormParams(rows, dim, eps),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 4, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], LAYER_NORM_KEY);
  }

  /**
   * rms_norm（最終次元・weight のみ）。layer_norm と同じ 1 行 = 1 workgroup の形で、
   * 縮約は二乗和 1 パス（ADR 0017）。
   */
  async #encodeRmsNorm(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const eps = rmsNormEps(step.node.attrs, `nodes (${step.node.op})`);
    const pipeline = await this.#state.cache.get(RMS_NORM_KEY, RMS_NORM_WGSL);
    const params = this.#writeParams(arena, rmsNormParams(rows, dim, eps), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 3, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], RMS_NORM_KEY);
  }

  /** softmax（最終次元、safe-softmax）。layer_norm と同じ 1 行 = 1 workgroup の形。 */
  async #encodeSoftmax(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const shape = step.outputShape;
    const dim = shape[shape.length - 1];
    const rows = numel(shape.slice(0, -1));
    const pipeline = await this.#state.cache.get(SOFTMAX_KEY, SOFTMAX_WGSL);
    const params = this.#writeParams(arena, softmaxParams(rows, dim), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      rows,
      1,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], SOFTMAX_KEY);
  }

  /**
   * 融合 attention（ADR 0023）。**1 ノード = 3 dispatch**（`cat` と同じ「複数 dispatch で
   * 1 ノード」の扱い。full-write は**ノードの出力 O について**成立する）:
   *
   * ① QK gemm（S を実体化・scale はタイル充填時に q/k 両方へ）→ ② 行統計（m と 1/Σexp）→
   * ③ PV gemm（A タイル充填時に `exp(S−m)·inv` を評価 = P 非実体化）。
   *
   * i8a8 変種（opt-in）では ① が 3 dispatch・③ が 3 dispatch に増える（最大 7）。**適格判定は
   * 段ごとに独立**（① は `D % 4 == 0`・③ は `N % 4 == 0`）なので、片方だけ i8a8 の**混成**が
   * 起こりうる — 満たさない段だけが f32 経路へ**沈黙で**縮退する（linear の `k % 4` と同じ
   * 流儀で、落ちたことは診断のパイプラインキーにだけ出る）。
   *
   * MUST: ① と ③ は 1 workgroup = 1 タイルなので、3 軸とも上限超過は fail loudly
   * （{@link tiledWorkgroups}）。縮退させるとタイルが欠落し、例外なしに O の一部が
   * 未書き込み（プール再利用なら前の値）で残る。② だけが行方向 grid-stride。
   * MUST: 一時バッファ（S / 行統計）は `retain(…, 0)` → ノード末尾で `release` する。
   * これでアリーナの参照計数が閉じ（`assertDrained`）、失敗経路でも `arena.destroy()` が
   * 拾う（確保と破棄を 1 箇所へ — ADR 0004）。
   */
  async #encodeAttention(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const [q, k, v] = step.inputShapes;
    // B と H は 1 本のバッチ軸へ畳む（契約は rank-4 head-first で、3 者の B / H は一致）。
    const batch = q[0] * q[1];
    const rows = q[2];
    const cols = k[2];
    const depth = q[3];
    const scale = attentionScale(step.node.attrs, `nodes (${step.node.op})`);
    const device = this.#state.gpu.device;
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

    // S[batch, M, N] と行統計 [batch·M, 2]。O は #encodeNode が確保済みなので、峰は
    // O + S + 統計（分解経路の S + P + 恒等 expand の 3 枚から 1 枚ぶん減る）。
    // f16 変種（`:c16` の array<f16> / s16 の pack2x16float）では S が半分のバイト数になる
    // （1024px の DiT で 1,073.7MB → 536.9MB）。
    const scores = arena.allocStorage(
      batch * rows * cols * (compute === "f16" ? 2 : scoreStorageBytes(scoreStorage)),
    );
    arena.retain(scores, 0);
    const stats = arena.allocStorage(batch * rows * ATTENTION_STATS_STRIDE * 4);
    arena.retain(stats, 0);

    // ① QK gemm — 縮約次元は D、出力の列は N。
    if (qkI8a8) {
      await this.#encodeAttentionQkI8a8(
        arena,
        inputs,
        scores,
        scoreStorage,
        { batch, rows, cols, depth },
        scale,
        where,
      );
    } else {
      const qkV4 = gemmUsesVec4(depth, cols);
      const qkKey = attentionQkKey(qkV4, compute, scoreStorage);
      const qkPipeline = await this.#state.cache.get(
        qkKey,
        attentionQkWgsl(qkV4, compute, scoreStorage),
      );
      const qkParams = this.#writeParams(
        arena,
        attentionQkParams(rows, cols, depth, scale),
        PARAMS_UNIFORM_USAGE,
      );
      const qkBindGroup = device.createBindGroup({
        layout: qkPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: qkParams } },
          { binding: 1, resource: { buffer: inputs[0] } },
          { binding: 2, resource: { buffer: inputs[1] } },
          { binding: 3, resource: { buffer: scores } },
        ],
      });
      this.#state.scheduler.dispatch(qkPipeline, qkBindGroup, [
        tiledWorkgroups(cols, GEMM_TILE, limit, `${where} ①QK`),
        tiledWorkgroups(rows, GEMM_TILE, limit, `${where} ①QK`),
        tiledWorkgroups(batch, 1, limit, `${where} ①QK`),
      ], qkKey);
    }

    // ② 行統計 — 1 行 = 1 workgroup で、行方向は grid-stride（softmax と同じ形）。
    // S を 1 回だけ読む regcache 変種は dim 依存の生成なので、`epc` がキー軸に増える
    // （値はどちらもビット同一 — src/kernels/attention.ts）。
    const statsRegCache = attentionStatsRegCache(cols);
    const statsKey = attentionStatsKey(compute, scoreStorage, statsRegCache);
    const statsPipeline = await this.#state.cache.get(
      statsKey,
      attentionStatsWgsl(compute, scoreStorage, statsRegCache),
    );
    const statsParams = this.#writeParams(
      arena,
      attentionStatsParams(batch * rows, cols),
      PARAMS_UNIFORM_USAGE,
    );
    const statsBindGroup = device.createBindGroup({
      layout: statsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: statsParams } },
        { binding: 1, resource: { buffer: scores } },
        { binding: 2, resource: { buffer: stats } },
      ],
    });
    this.#state.scheduler.dispatch(statsPipeline, statsBindGroup, [
      gridStrideWorkgroups(batch * rows, 1, limit),
      1,
      1,
    ], statsKey);

    // ③ PV gemm — 縮約次元は N、出力の列は D。
    if (pvI8a8) {
      await this.#encodeAttentionPvI8a8(
        arena,
        inputs,
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
      const pvPipeline = await this.#state.cache.get(
        pvKey,
        attentionPvWgsl(pvV4, compute, scoreStorage),
      );
      const pvParams = this.#writeParams(
        arena,
        attentionPvParams(rows, depth, cols),
        PARAMS_UNIFORM_USAGE,
      );
      const pvBindGroup = device.createBindGroup({
        layout: pvPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: pvParams } },
          { binding: 1, resource: { buffer: scores } },
          { binding: 2, resource: { buffer: inputs[2] } },
          { binding: 3, resource: { buffer: stats } },
          { binding: 4, resource: { buffer: out } },
        ],
      });
      this.#state.scheduler.dispatch(pvPipeline, pvBindGroup, [
        tiledWorkgroups(depth, GEMM_TILE, limit, `${where} ③PV`),
        tiledWorkgroups(rows, GEMM_TILE, limit, `${where} ③PV`),
        tiledWorkgroups(batch, 1, limit, `${where} ③PV`),
      ], pvKey);
    }

    // MUST: ノード境界で一時バッファを返す（アリーナの不変条件 — 抜けると assertDrained で
    // 落ちるか、プール再利用から外れて peak が過大に出る）。
    arena.release(stats);
    arena.release(scores);
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
   * MUST: 一時バッファ（`qq` / `qs` / `kq` / `ks`）は `retain(…, 0)` → 末尾で `release`。
   * MUST: `D` のオーバフロー門は fail loudly（黙って通すと i32 の巻き戻りで符号ごと化ける。
   * 実測形の D ≤ 384 に対して門は桁で余裕があるが、置かないと退行の受け皿が消える）。
   */
  async #encodeAttentionQkI8a8(
    arena: RunArena,
    inputs: readonly GPUBuffer[],
    scores: GPUBuffer,
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
    const device = this.#state.gpu.device;
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // 量子化した q / k（i8 を 4 詰め）と per-token scale。ノード内で閉じた一時領域で、
    // q 側は**行** scale・k 側は**出力列**の scale になる（同じ per-token 量子化の別の読み方）。
    const qq = arena.allocStorage(Math.max(4, batch * rows * depth));
    arena.retain(qq, 0);
    const qs = arena.allocStorage(Math.max(4, batch * rows * 4));
    arena.retain(qs, 0);
    const kq = arena.allocStorage(Math.max(4, batch * cols * depth));
    arena.retain(kq, 0);
    const ks = arena.allocStorage(Math.max(4, batch * cols * 4));
    arena.retain(ks, 0);

    // (a)(b) 活性の per-token 量子化（1 行 = 1 workgroup・行方向 grid-stride）
    const quantizePipeline = await this.#state.cache.get(QUANTIZE_ROWS_KEY, QUANTIZE_ROWS_WGSL);
    const quantize = (
      source: GPUBuffer,
      payload: GPUBuffer,
      scales: GPUBuffer,
      count: number,
    ): void => {
      const params = this.#writeParams(
        arena,
        quantizeRowsParams(count, depth),
        PARAMS_UNIFORM_USAGE,
      );
      const bindGroup = device.createBindGroup({
        layout: quantizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: source } },
          { binding: 2, resource: { buffer: payload } },
          { binding: 3, resource: { buffer: scales } },
        ],
      });
      this.#state.scheduler.dispatch(quantizePipeline, bindGroup, [
        gridStrideWorkgroups(count, 1, limit),
        1,
        1,
      ], QUANTIZE_ROWS_KEY);
    };
    quantize(inputs[0], qq, qs, batch * rows);
    quantize(inputs[1], kq, ks, batch * cols);

    // (c) 整数内積の GEMM（半スケールは dequant 側で q / k の両方へ — 設計 §2.1）。
    // 幾何は ③PV と**別に**選ぶ（③ だけ N = D の 1 タイル化が勝つ — 実測）。
    const v4 = attentionQkI8a8UsesVec4(cols);
    const dp4a = this.#state.i8a8Dot === "dp4a";
    const geometry = defaultI8a8Geometry("attention_qk");
    const key = attentionQkI8a8Key(v4, dp4a, scoreStorage, geometry);
    const pipeline = await this.#state.cache.get(
      key,
      attentionQkI8a8Wgsl(v4, dp4a, scoreStorage, geometry),
    );
    const params = this.#writeParams(
      arena,
      attentionQkI8a8Params(rows, cols, depth, scale),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: qq } },
        { binding: 2, resource: { buffer: kq } },
        { binding: 3, resource: { buffer: scores } },
        { binding: ATTENTION_QK_Q_SCALE_BINDING, resource: { buffer: qs } },
        { binding: ATTENTION_QK_K_SCALE_BINDING, resource: { buffer: ks } },
      ],
    });
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(cols, i8a8TileN(geometry), limit, `${where} ①QK i8a8`),
      tiledWorkgroups(rows, i8a8TileM(geometry), limit, `${where} ①QK i8a8`),
      tiledWorkgroups(batch, 1, limit, `${where} ①QK i8a8`),
    ], key);

    // MUST: ノード境界で一時バッファを返す（確保と破棄を 1 箇所へ — ADR 0004）。
    arena.release(ks);
    arena.release(kq);
    arena.release(qs);
    arena.release(qq);
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
   * MUST: 一時バッファ（`vt` / `vq` / `vs`）は `retain(…, 0)` → 末尾で `release`。
   * MUST: `N` のオーバフロー門は fail loudly（|acc| ≤ N·127²。実測形の最大 N = 16,384 に対し
   * 門は桁で余裕があるが、置かないと退行の受け皿が消える）。
   */
  async #encodeAttentionPvI8a8(
    arena: RunArena,
    inputs: readonly GPUBuffer[],
    scores: GPUBuffer,
    scoreStorage: ScoreStorage,
    stats: GPUBuffer,
    out: GPUBuffer,
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
    const device = this.#state.gpu.device;
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;

    // Vᵀ（f32・permute の実体化）と、その量子化結果。どれもノード内で閉じた一時領域。
    const vt = arena.allocStorage(Math.max(4, batch * depth * cols * 4));
    arena.retain(vt, 0);
    const vq = arena.allocStorage(Math.max(4, batch * depth * cols));
    arena.retain(vq, 0);
    const vs = arena.allocStorage(Math.max(4, batch * depth * 4));
    arena.retain(vs, 0);

    // (a) v[B·H,N,D] → Vᵀ[B·H,D,N]（既存の strided 読みコピー族 — permute そのもの）。
    // MUST: stride は**入力** `[B·H,N,D]` の連続 stride から組む（出力 shape から組むと
    // D == N のときだけ一致する。実測形は D != N なので露見するが、単体テストが本来の検出器）。
    const stridedSpec = { dtype: "f32" } as const;
    const permuteKey = stridedKey(stridedSpec);
    const permutePipeline = await this.#state.cache.get(permuteKey, stridedWgsl(stridedSpec));
    const permuteParams = this.#writeParams(
      arena,
      stridedParams(
        [batch, depth, cols],
        permuteSrcStrides([batch, cols, depth], [0, 2, 1]),
        0,
      ),
      PARAMS_STORAGE_USAGE,
    );
    const permuteBindGroup = device.createBindGroup({
      layout: permutePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: permuteParams } },
        { binding: 1, resource: { buffer: inputs[2] } },
        { binding: 2, resource: { buffer: vt } },
      ],
    });
    this.#state.scheduler.dispatch(permutePipeline, permuteBindGroup, [
      gridStrideWorkgroups(batch * depth * cols, STRIDED_WORKGROUP_SIZE, limit),
      1,
      1,
    ], permuteKey);

    // (b) Vᵀ の量子化（行 = (b,h,d)・行長 N — per-column scale と N 連続パックが同時に出る）
    const quantizePipeline = await this.#state.cache.get(QUANTIZE_ROWS_KEY, QUANTIZE_ROWS_WGSL);
    const quantizeParams = this.#writeParams(
      arena,
      quantizeRowsParams(batch * depth, cols),
      PARAMS_UNIFORM_USAGE,
    );
    const quantizeBindGroup = device.createBindGroup({
      layout: quantizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: quantizeParams } },
        { binding: 1, resource: { buffer: vt } },
        { binding: 2, resource: { buffer: vq } },
        { binding: 3, resource: { buffer: vs } },
      ],
    });
    this.#state.scheduler.dispatch(quantizePipeline, quantizeBindGroup, [
      gridStrideWorkgroups(batch * depth, 1, limit),
      1,
      1,
    ], QUANTIZE_ROWS_KEY);

    // (c) 整数内積の GEMM（P̃ は A タイル充填で作る = 非実体化のまま）
    const v4 = attentionPvI8a8UsesVec4(depth);
    const dp4a = this.#state.i8a8Dot === "dp4a";
    const geometry = defaultI8a8Geometry("attention_pv");
    const key = attentionPvI8a8Key(v4, dp4a, scoreStorage, geometry);
    const pipeline = await this.#state.cache.get(
      key,
      attentionPvI8a8Wgsl(v4, dp4a, scoreStorage, geometry),
    );
    const params = this.#writeParams(
      arena,
      attentionPvI8a8Params(rows, depth, cols),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: scores } },
        { binding: 2, resource: { buffer: vq } },
        { binding: 3, resource: { buffer: stats } },
        { binding: 4, resource: { buffer: out } },
        { binding: ATTENTION_PV_V_SCALE_BINDING, resource: { buffer: vs } },
      ],
    });
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(depth, i8a8TileN(geometry), limit, `${where} ③PV i8a8`),
      tiledWorkgroups(rows, i8a8TileM(geometry), limit, `${where} ③PV i8a8`),
      tiledWorkgroups(batch, 1, limit, `${where} ③PV i8a8`),
    ], key);

    // MUST: ノード境界で一時バッファを返す（確保と破棄を 1 箇所へ — ADR 0004）。
    arena.release(vs);
    arena.release(vq);
    arena.release(vt);
  }

  /**
   * embedding（行 gather）。範囲外添字の扱いは src/kernels/embedding.ts の裁定
   * （GPU は NaN 汚染 / CPU 参照は throw）。attrs の padding_idx は forward に効かないので
   * カーネルへ渡さない。
   */
  async #encodeEmbedding(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const weight = step.inputShapes[0];
    const count = numel(step.outputShape);
    const weightStorage = this.#weightStorage(step);
    const key = embeddingKey(weightStorage);
    const pipeline = await this.#state.cache.get(key, embeddingWgsl(weightStorage));
    const params = this.#writeParams(
      arena,
      embeddingParams(count, weight[1], weight[0]),
      PARAMS_UNIFORM_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: inputs[1] } },
        { binding: 3, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, weightStorage, EMBEDDING_SCALE_BINDING),
      ],
    });
    const groups = gridStrideWorkgroups(
      count,
      EMBEDDING_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
  }

  /**
   * masked_fill。出力と x は同形・連続で、mask だけを右詰め broadcast の stride で読む。
   * stride の組み立ては strided 族の expand と同じ規則（{@link expandSrcStrides}）。
   */
  async #encodeMaskedFill(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const outShape = step.outputShape;
    // 規則は expand と同一だが、診断の主語は masked_fill の側に付け替える（グラフに expand が
    // 無いのに「expand の入力」と出ると原因の当たりを外す）。
    const maskStrides = expandSrcStrides(step.inputShapes[1], outShape, {
      src: "masked_fill の mask",
      out: "masked_fill の出力",
    });
    const value = maskedFillValue(step.node.attrs, `nodes (${step.node.op})`);
    const pipeline = await this.#state.cache.get(MASKED_FILL_KEY, MASKED_FILL_WGSL);
    const params = this.#writeParams(
      arena,
      maskedFillParams(outShape, maskStrides, value),
      PARAMS_STORAGE_USAGE,
    );
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: inputs[0] } },
        { binding: 2, resource: { buffer: inputs[1] } },
        { binding: 3, resource: { buffer: out } },
      ],
    });
    const groups = gridStrideWorkgroups(
      numel(outShape),
      MASKED_FILL_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], MASKED_FILL_KEY);
  }

  /** conv1d（直接畳み込み、groups / dilation は attrs）。1 スレッド = 1 出力要素の grid-stride。 */
  async #encodeConv1d(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding, dilation, groups } = conv1dAttrs(
      step.node.attrs,
      `nodes (${step.node.op})`,
    );
    const weightStorage = this.#weightStorage(step);
    const key = conv1dKey(weightStorage);
    const pipeline = await this.#state.cache.get(key, conv1dWgsl(weightStorage));
    const params = this.#writeParams(
      arena,
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
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 4, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, weightStorage, CONV1D_SCALE_BINDING),
      ],
    });
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV1D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [workgroups, 1, 1], key);
  }

  /**
   * conv2d（stride / padding / dilation は H/W の 2 成分）。**groups で 2 カーネルを踏み分ける**
   * （ADR 0024）: `groups == 1` は implicit GEMM、`groups > 1` は直接畳み込み。
   *
   * MUST: Kh / Kw は**重みの第 3 / 第 4 軸**をこの順で読む。入れ替えても正方カーネルでは
   * 数値が一致するので、テストは Kh ≠ Kw の形で固定する（src/kernels/conv2d.ts）。
   */
  async #encodeConv2d(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
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
      await this.#encodeConv2dIgemm(step, inputs, out, arena, dims, weightStorage);
      return;
    }
    const key = conv2dKey(weightStorage);
    const pipeline = await this.#state.cache.get(key, conv2dWgsl(weightStorage));
    const params = this.#writeParams(arena, conv2dParams(dims), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 4, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, weightStorage, CONV2D_SCALE_BINDING),
      ],
    });
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV2D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [workgroups, 1, 1], key);
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
  async #encodeConv2dIgemm(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
    dims: Conv2dDims,
    weightStorage: WeightStorage,
  ): Promise<void> {
    const m = dims.channelsOut;
    const n = dims.heightOut * dims.widthOut;
    const kFlat = dims.channelsIn * dims.kernelH * dims.kernelW;
    const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
    const mTile = conv2dIgemmMTile(m);
    const key = conv2dIgemmKey(weightStorage, v4, mTile);
    const pipeline = await this.#state.cache.get(key, conv2dIgemmWgsl(weightStorage, v4, mTile));
    const params = this.#writeParams(arena, conv2dIgemmParams(dims), PARAMS_UNIFORM_USAGE);
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 4, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, weightStorage, CONV2D_SCALE_BINDING),
      ],
    });
    const limit = this.#state.gpu.limits.maxComputeWorkgroupsPerDimension;
    const where = `conv2d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
    this.#state.scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(n, GEMM_TILE, limit, where),
      tiledWorkgroups(m, mTile, limit, where),
      tiledWorkgroups(dims.batch, 1, limit, where),
    ], key);
  }

  /**
   * conv_transpose1d（gather 形）。1 スレッド = 1 出力要素で出力全域を書く（ADR 0014）。
   *
   * MUST: Cin は**重みの第 1 軸**（`[Cin, Cout, K]`）。x[1] と一致することは契約検査済みだが、
   * ここで weight[0] を使うのは「転置レイアウトの正本は重み側」という読みを崩さないため。
   */
  async #encodeConvTranspose1d(
    step: NodePlan,
    inputs: readonly GPUBuffer[],
    out: GPUBuffer,
    arena: RunArena,
  ): Promise<void> {
    const [x, weight] = step.inputShapes;
    const outShape = step.outputShape;
    const { stride, padding } = convTranspose1dAttrs(step.node.attrs, `nodes (${step.node.op})`);
    const weightStorage = this.#weightStorage(step);
    const key = convTranspose1dKey(weightStorage);
    const pipeline = await this.#state.cache.get(key, convTranspose1dWgsl(weightStorage));
    const params = this.#writeParams(
      arena,
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
    const bindGroup = this.#state.gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        ...inputs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
        { binding: 4, resource: { buffer: out } },
        ...this.#weightScaleEntries(step, weightStorage, CONV_TRANSPOSE1D_SCALE_BINDING),
      ],
    });
    const workgroups = gridStrideWorkgroups(
      numel(outShape),
      CONV_TRANSPOSE1D_WORKGROUP_SIZE,
      this.#state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    this.#state.scheduler.dispatch(pipeline, bindGroup, [workgroups, 1, 1], key);
  }

  #writeParams(arena: RunArena, params: Uint32Array<ArrayBuffer>, usage: number): GPUBuffer {
    const buffer = arena.allocHostWritten(params.byteLength, usage);
    this.#state.gpu.device.queue.writeBuffer(buffer, 0, params);
    return buffer;
  }

  /** MUST: 読み戻すのはグラフ出力のみ。中間値はプール再利用で内容が入れ替わっている。 */
  async #readOutputs(
    env: ReadonlyMap<string, GPUBuffer>,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
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
        const buffer = env.get(name);
        if (buffer === undefined) throw new ExecutionError(`グラフ出力 '${name}' のバッファが無い`);
        if (!arena.isReadable(buffer)) {
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
