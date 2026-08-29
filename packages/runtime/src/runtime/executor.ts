/**
 * 最小のグラフ実行器。
 *
 * 構造: 「計画（純関数・plan.ts）→ **導出**（レシピ構築 — src/runtime/recipe-builder.ts）→ **実行**
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
 * MUST: バッファ破棄の前に未 submit のエンコードを必ず片付ける — 成功経路は submit（待ちが要る
 * なら flush、待ちを readback の `mapAsync` へ集約した run は `submitPending`）、失敗経路は
 * discard（discard-or-flush before destroy）。破棄済みバッファを参照するエンコードを submit
 * するとコマンドバッファ丸ごと失敗し、同じスケジューラに相乗りしている無関係な dispatch まで
 * 実行されないまま誤った値が静かに残る。
 */

import {
  assertRuntimeSupport,
  createShardValidator,
  extractIrGraph,
  IR_METADATA_KEY,
  type KarumeModel,
  type ReadyInitializer,
} from "../format/container.ts";
import { alignF16Payload, decodeF16 } from "../format/f16.ts";
import { decodeI4 } from "../format/i4.ts";
import { alignI8Payload, decodeI8 } from "../format/i8.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";
import { parseSafetensors, type SafetensorsFile, tensorBytes } from "../format/safetensors.ts";
import { type ArenaStats, RunArena, STORAGE_USAGE } from "../gpu/arena.ts";
import {
  type AttentionI8a8Decision,
  decideAttentionI8a8Dot,
  formatAttentionI8a8Decision,
} from "../gpu/attention-dp4a-canary.ts";
import {
  BatchScopeError,
  discardFailureScopes,
  type GpuContext,
  popFailureScopes,
  pushFailureScopes,
  ResidentTensor,
  ResidentTensorError,
  RUNTIME_INTERNAL,
} from "../gpu/device.ts";
import { PipelineCache } from "../gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../gpu/submit.ts";
import { BUFFER_USAGE, MAP_MODE } from "../gpu/webgpu-constants.ts";
import { dp4aAvailable } from "../kernels/linear-i8a8.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
/** S の格納形（{@link SessionOptions.attentionScoreStorage} — 公開面で名前を持てるように再輸出）。 */
export type { ScoreStorage } from "../kernels/score-storage.ts";
import { numel, RUNTIME_SUPPORT } from "../ops.ts";
import { type AdmissionReport, estimateGraphMemory, type EstimateOptions } from "./estimate.ts";
import { type ExecStep, type FusionCounts, planFusions } from "./fusion.ts";
import { GenerationContext, type GenerationContextHost } from "./generation-context.ts";
/**
 * 1 生成ぶんの可変 state の所有者（ADR 0066 決定 1）。**型としてのみ**公開する — 入口は
 * {@link Session.createGenerationContext} だけで、直接構築すると errorScope の門（確保失敗の
 * 検出）と容量ゲート（追記 5）を迂回できてしまう（`GpuContext` / `ResidentTensor` と同じ流儀）。
 */
export type { GenerationContext } from "./generation-context.ts";
/**
 * ルール別の融合適用回数（{@link SessionDiagnostics.lastRunFusions} の型 — 公開面で名前を
 * 持てるように再輸出。`ScoreStorage` と同じ扱いで、素通し再輸出ではなく**既に公開している型の
 * 構成要素**を面に載せるだけ）。
 *
 * NOTE: キー集合は融合ルールそのものなので、ルールを 1 本足せばキーが増える（型としては
 * 破壊的変更になりうる）。現状でも `SessionDiagnostics` 経由で構造的に露出しているため、
 * 名前が付くことで実質的な互換性の面は変わらない。
 */
export type { FusionCounts } from "./fusion.ts";
import {
  assertGenerationBindings,
  bindSymbols,
  countUses,
  declaredDtypes,
  ExecutionError,
  planGraph,
  statesOnlySymbols,
  type SymbolBindings,
  validateGraphContracts,
} from "./plan.ts";
import {
  assertGenerationRun,
  bakeBindGroups,
  type BakedGroups,
  bakeGenerationBindGroups,
  derivePlanSlots,
  executeBakedPlan,
  executeStepRecipe,
  type GenerationEncoding,
  type GenerationLimits,
  type PlanSlots,
  type StepRecipe,
} from "./recipe.ts";
import { RecipeBuilder } from "./recipe-builder.ts";
import {
  planWeightResidency,
  type ResidentWeight,
  type WeightResidency,
} from "./weight-residency.ts";
import {
  type ComputePrecision,
  type EnqueueOptions,
  type GenerationContextSpec,
  I8A8_DOT,
  type I8a8Dot,
  type ParamsCacheStats,
  type PreparedPlanStats,
  ROW_BLOCK_SPLIT,
  type RunInput,
  type RunInputs,
  type RunOutputs,
  type SessionBuildStats,
  type SessionDiagnostics,
  type SessionOptions,
  type StorageDiagnostics,
  type Tensor,
} from "./session-types.ts";
export type {
  ComputePrecision,
  EnqueueOptions,
  GenerationContextSpec,
  I8a8Dot,
  ParamsCacheStats,
  PlanBackingStats,
  PreparedPlanStats,
  RunInput,
  RunInputs,
  RunOutputs,
  SessionBuildStats,
  SessionDiagnostics,
  SessionOptions,
  StateBackingStats,
  StorageDiagnostics,
  Tensor,
} from "./session-types.ts";
export { I8A8_DOT, ROW_BLOCK_SPLIT } from "./session-types.ts";

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

/** MUST: `queue.writeBuffer` で書くバッファはプール外（アリーナの不変条件）。 */
const HOST_WRITTEN_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST |
  BUFFER_USAGE.COPY_SRC;

/**
 * 発行時点で固定した入力の束（{@link captureInputs} の成果物）。実行本体が読むのは**この写し
 * だけ**で、利用者の `inputs` Record は 1 度も引き直さない。
 */
type CapturedInputs = {
  /**
   * 名前 → 入力の実体。**Record の member identity をここで固定する**のが目的で、器が `Map`
   * なのは入力名 "__proto__" でも素直に引けるため（`inputs[name]` の索引はプロトタイプ由来の
   * 値を拾いうる）。
   */
  readonly values: ReadonlyMap<string, RunInput>;
  /** ホスト配列入力の shape の**複製**（束縛解決 = {@link bindSymbols} の唯一の材料）。 */
  readonly inputShapes: Record<string, readonly number[]>;
  /** 常駐入力だけの表（束縛予約・計画キー・`deferredInputs`）。 */
  readonly residentInputs: ReadonlyMap<string, ResidentTensor>;
};

/**
 * 入力の束を発行時点で写し取り、「ホスト配列の shape 表」と「常駐入力の表」に分ける。
 *
 * MUST: 呼ぶのは {@link Session.run} / {@link Session.enqueue} の**同期区間**。実行本体は
 * `#serialize` 越し（1 マイクロタスク以降）に走るので、ここで写さないと「発行直後に呼び出し側が
 * 触った Record / shape 配列」が本体から見えてしまう — member 差し替えは沈黙誤値、shape の
 * 書き換えは「たまたま fail loudly」になる（どちらも run の JSDoc が勧める非 await 並行発行で
 * 素直に踏める形）。写すのは安価な metadata だけで、`Tensor.data`（TypedArray の実体）は
 * **借りたまま** — 契約は run の「入力の寿命」節。
 * MUST: 利用者の inputs 由来のキーを蓄積する器は null プロトタイプ（理由は plan.ts の
 * bindSymbols と同じ）。素の `{}` では入力名が "__proto__" のとき shape 配列が [[Prototype]] に
 * 化け、own property が作られないまま「入力が渡されていない」で落ちる。常駐入力側は `Map` な
 * ので同じ穴が無い。
 */
const captureInputs = (inputs: RunInputs): CapturedInputs => {
  const values = new Map<string, RunInput>();
  const inputShapes: Record<string, readonly number[]> = Object.create(null);
  const residentInputs = new Map<string, ResidentTensor>();
  for (const [name, value] of Object.entries(inputs)) {
    values.set(name, value);
    if (value instanceof ResidentTensor) residentInputs.set(name, value);
    // MUST: shape は**複製**して持つ（借りたままだと、発行直後の書き換えが束縛解決の材料を
    // 変える）。常駐入力はホスト側に shape を持たない寿命クラスなので写す対象が無い。
    else inputShapes[name] = [...value.shape];
  }
  return { values, inputShapes, residentInputs };
};

/** 常駐入力の名前集合（{@link bindSymbols} の `deferredInputs`）。空なら渡さない。 */
const residentNames = (
  residentInputs: ReadonlyMap<string, ResidentTensor>,
): ReadonlySet<string> | undefined =>
  residentInputs.size === 0 ? undefined : new Set(residentInputs.keys());

/**
 * 常駐テンソルを束ねてよいかの門（破棄済み / 所有 GpuContext 違い）。
 *
 * MUST: 破棄済みバッファの束縛は createBindGroup の validation 失敗になり、例外にならないまま
 * dispatch が no-op 化して出力が全て 0 になる（ここが唯一の同期的な検出点）。
 * MUST: 所有 GpuContext も照合する。resident の識別子は GpuContext ごとの独立採番で、
 * 導出済み計画のキー / backing signature に載るのはその数値だけなので、別 context の同 id・
 * 同サイズな resident は**キーが衝突する**。ヒット run（焼き込み済み backing）は渡された実体を
 * 一切参照しないため、例外も警告も無く前の context の古い値を読む（ミス経路だけが device
 * 不一致の validation で偶然落ちる = キャッシュが当たった瞬間に検出が消える）。
 * `BatchScope` が owner を検査しているのと同じ門をここに置く。
 */
const assertResidentUsable = (
  resident: ResidentTensor,
  gpu: GpuContext,
  where: string,
): void => {
  if (resident.disposed) {
    throw new ExecutionError(`${where}: 破棄済みの常駐テンソル '${resident.label}' は使えない`);
  }
  if (resident[RUNTIME_INTERNAL].owner !== gpu) {
    throw new ResidentTensorError(
      `${where}: 常駐テンソル '${resident.label}' は別の GpuContext が所有している` +
        "（識別子は GpuContext ごとの独立採番なので、跨いで渡すと導出済み計画のキーが衝突する）",
    );
  }
};

/**
 * context 所有の実体と論理長を run 1 回ぶんの束にする（{@link GenerationEncoding}）。
 *
 * MUST: 論理長は**呼び出し側が 1 度読んだ値**を受け取る（ここで `context.pastLength` を読み直さ
 * ない）。uniform へ書く値・dispatch 数の算出・容量の検査が同じ 1 つの値から出ることが、
 * 「GPU が走査する範囲」と「ホストが撃った workgroup 数」の一致の根拠。
 */
const generationEncoding = (
  context: GenerationContext,
  past: number,
  query: number,
): GenerationEncoding => ({
  slots: new Map(
    [...context[RUNTIME_INTERNAL].slots].map(([name, slot]) => [name, slot.buffer]),
  ),
  lengths: context[RUNTIME_INTERNAL].lengths,
  past,
  query,
});

/**
 * 進行中 run の常駐入力束縛を全て返す（{@link ResidentTensor.dispose} の予約を解く）。
 *
 * MUST: 成功・失敗のどちらの経路でも必ず 1 度呼ぶ。返し損ねるとその常駐テンソルは
 * Session の寿命いっぱい破棄できなくなり、二度返すと簿記の破れとして fail loudly になる。
 */
const releaseBoundResidents = (bound: ResidentTensor[]): void => {
  for (const resident of bound) resident[RUNTIME_INTERNAL].releaseBound();
  bound.length = 0;
};

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
 * i8 / i4 格納の companion scale テンソル（ADR 0019 / 0069）。実在・F32・形・co-shard は
 * shard 進行検証（format/container.ts）が済ませているので、ここは view を組むだけ。
 *
 * MUST: `Float32Array` の view はコピーせずに張る（scale は重み本体に比べれば小さいが、
 * ここで無条件コピーを挟むと「生バイトのまま常駐」の経路が二重確保になる）。絶対 offset の
 * 4 バイト整列は safetensors リーダが保証済み。
 */
const scaleTensor = (
  item: ReadyInitializer,
  storage: string,
): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
} => {
  const view = item.scale;
  if (view === undefined) {
    // 存在は型の上でだけ optional なので、黙って読み飛ばさず言い直す（fail loudly）。
    throw new ExecutionError(
      `initializer '${item.name}': 格納 ${storage} なのに storage.scale が無い`,
    );
  }
  return {
    bytes: tensorBytes(item.file, view),
    values: new Float32Array(item.file.buffer, view.byteOffset, view.byteLength / 4),
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
 *
 * NOTE: 軸が決まらない形（消費側が食い違う / 軸の定義が無い）はプランナ
 * （{@link planWeightResidency}）が先に落とすので、ここは `number` を受ける。
 */
const assertChannelScale = (
  name: string,
  weightShape: readonly number[],
  scaleShape: readonly number[],
  axis: number,
): void => {
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
 * shard 逐次面（{@link createSessionFromShards}）が 1 本ずつ受け取る shard。
 *
 * hub の `StreamedAsset`（`id` = manifest の path）と**構造互換**の型を runtime 側で独立に
 * 持つ — runtime → hub の依存を作らずに、配布形のファイル名を失敗の帰属先へ通すため。
 */
export type ModelShard = {
  /**
   * 資産の実名（hub 経由なら manifest の path）。
   *
   * MUST: 失敗とフェンスの帰属はこの id を名乗る。届いた順の連番だけでは「配布形のどの
   * ファイルが壊れているか」が呼び手にも利用者にも決まらない（列の組み方は呼び手側にあり、
   * 連番は runtime から見た到着順でしかない）。
   */
  readonly id: string;
  /** shard のバイト列。buffer 全体を占める view MUST（slice で辻褄を合わせると RAM ピークが倍増する）。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
};

/**
 * Session 構築（{@link Session.build}）が消費する shard 1 本。`origin` はエラーとフェンスの
 * 帰属先で、**全量面は undefined**（帰属先が 1 つしかない単一ファイル面の文言を変えない
 * MUST — ADR 0070 受入①の契約。合成 id を作ると shard 面の語彙が全量面へ漏れる）。
 */
type WeightShard = {
  readonly file: SafetensorsFile;
  readonly origin: string | undefined;
};

/** 失敗・フェンスの帰属先。連番は到着順の補助で、実名（{@link ModelShard.id}）が本体。 */
const shardOrigin = (index: number, id: string): string => `shard [${index}] '${id}'`;

/**
 * 重みアップロード区間のラベル（errorScope とフェンスの帰属先）。`origin` から導出する —
 * shard 側と別々に持つと、片方だけ名乗り方が変わったときに 2 つの名前で同じ失敗が出る。
 */
const uploadLabel = (origin: string | undefined): string =>
  origin === undefined ? "重みのアップロード" : `${origin} の重みアップロード`;

/**
 * shard 由来の失敗に帰属先を足して**同じエラーを返す**（`origin` が無い全量面は素通し —
 * 文言が 1 文字も変わらない）。
 *
 * MUST: 新しい Error で包み直さない。呼び出し側はクラスで分岐しており（宣言違反 =
 * `ContainerError` / パーサ門 = `SafetensorsError`）、包むと分岐が壊れて stack も切れる。
 */
const attributeToShard = (origin: string | undefined, cause: unknown): unknown => {
  if (origin !== undefined && cause instanceof Error) {
    cause.message = `${origin}: ${cause.message}`;
  }
  return cause;
};

/**
 * shard 面の消費列: 検証済みのグラフ shard を先頭に、残り shard を parse して流す。
 *
 * MUST: グラフ shard は最初の 1 本だけ（ADR 0070 決定 3）。後続に `karume_ir` 持ちが
 * 現れたら取り違え（別モデルの混入・並び順の崩れ）の徴候なので fail loudly。
 * NOTE: グラフ shard のバイト列は {@link PreparedModel} が持ち主で、この generator の寿命では
 * 手放せない（2 段境界の代償 — ADR 0070 決定 3 がグラフ shard を「karume_ir + 小テンソル」と
 * 規定しているので、RAM ピーク目標「O(最大**重み** shard)」は崩れない）。
 * NOTE: 後続の連番は 1 から振る（グラフ shard が [0] — 帰属ラベルの通し番号は 2 段境界の
 * 前後で変わらない）。
 */
const followingShards = async function* (
  graphShard: WeightShard,
  iterator: AsyncIterator<ModelShard>,
): AsyncGenerator<WeightShard, void, unknown> {
  yield graphShard;
  let index = 1;
  while (true) {
    const next = await iterator.next();
    if (next.done === true) return;
    const origin = shardOrigin(index, next.value.id);
    const file = parseShard(next.value.bytes, origin);
    if (file.metadata.has(IR_METADATA_KEY)) {
      throw new ExecutionError(
        `${origin}: __metadata__.${IR_METADATA_KEY} を持つグラフ shard が複数ある` +
          "（グラフ shard は最初の 1 本だけ — ADR 0070 決定 3）",
      );
    }
    yield { file, origin };
    index += 1;
  }
};

/**
 * shard のバイト列を parse する。bytes が buffer 全体を占めることの確認は ADR 0038 §5 の
 * `openModel` への受け渡しと同じ規律（slice で辻褄を合わせると RAM ピークが倍増する）。
 *
 * 非 tight view もパーサ門（`SafetensorsError`）も**その shard を名乗って**落ちる —
 * 壊れた 1 本を配布形から特定するのに要るのは連番ではなくファイル名。
 */
const parseShard = (bytes: Uint8Array<ArrayBuffer>, origin: string): SafetensorsFile => {
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new ExecutionError(
      `${origin}: bytes が buffer 全体を占めていない（byteOffset ${bytes.byteOffset} / ` +
        `byteLength ${bytes.byteLength} / buffer ${bytes.buffer.byteLength}）`,
    );
  }
  try {
    return parseSafetensors(bytes.buffer);
  } catch (cause) {
    throw attributeToShard(origin, cause);
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
  /**
   * generation run の run 前検査に要る計画事実（{@link assertGenerationRun}）。state ノードを
   * 持たないグラフでは空で、その run は検査を 1 つも通さない（見る対象が無い）。
   */
  readonly generation: GenerationLimits;
};

/**
 * generation run 1 回ぶんの指定（{@link Session.run} の第 3 引数）。
 *
 * `queryLength` は今 step の実 token 数（prefill は `1..chunkLength`・decode は 1）で、
 * **`pastLength` は渡さない** — 論理長の進行は context が所有し、run の成功でのみ進む
 * （ADR 0066 決定 6 の二重簿記の禁止）。
 */
export type GenerationRun = {
  readonly context: GenerationContext;
  readonly queryLength: number;
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
   * **この backing 実体の世代識別子**（`Session.#backingBuilds` の採番 — 単調増加で再利用しない）。
   *
   * MUST: 同じ `key` で作り直した backing は必ず別の値になる。GenerationContext が焼いた
   * bind group（ADR 0066 決定 5）はこの実体の slot / 入力バッファを掴んでいるので、識別子を
   * key で代用すると「退役して作り直した backing」の切替が検出できず、破棄済みバッファを
   * 束ねた古い group がそのまま dispatch される（例外も警告も出ない）。
   */
  readonly build: number;
  /**
   * transient slot の常駐バイト数（**入力バッファは含まない**）。
   * MUST: この定義を変えない — 「slot 表の総バイト数 = 非 backed run のプール確保」という
   * footprint 不変の門（tests/gpu_plan_backing_test.ts）がこの値そのもので、入力ぶんを混ぜると
   * 門が観測しているものが変わる（入力は signature ごとに固定サイズで、slot に比べれば桁が違う）。
   */
  readonly bytes: number;
  /**
   * グラフ入力名 → 束ねる実体。通常入力は backing 所有の常駐バッファ（`HOST_WRITTEN_USAGE`・
   * サイズは解決済み shape から確定するので同一 signature では作り直す理由が無い）、常駐入力は
   * {@link ResidentTensor} の実体そのもの（**backing の所有外**）。
   */
  readonly inputs: ReadonlyMap<string, GPUBuffer>;
  /**
   * この backing が焼き込みで参照している常駐入力。構築時に retain し、退役時に release する
   * （参照中の {@link ResidentTensor.dispose} を fail loudly にする根拠）。
   */
  readonly residents: readonly ResidentTensor[];
  /**
   * 焼き込み済み bind group（{@link bakeBindGroups}）。backed run はこれを dispatch するだけ。
   * state を束ねる位置は `undefined` の穴で、埋めるのは context 側の焼き込み（決定 5）。
   */
  readonly groups: BakedGroups;
  /**
   * slot 表と slot 添字 → 常駐バッファ。**context 側の焼き込みが同じ値解決を再現する**ための
   * 材料で、Session 側の焼き込み（構築時 1 度）にはもう要らない。
   * MUST: 焼き込みに使ったものをそのまま持つ（作り直して渡すと、2 つの焼き込みが別々の実体を
   * 束ねうる）。
   */
  readonly slots: PlanSlots;
  readonly buffers: readonly GPUBuffer[];
  /** グラフ出力名 → 実体（読み戻し先 — 構築時に確定。backed run は env を組まない）。 */
  readonly outputs: ReadonlyMap<string, GPUBuffer>;
  /**
   * backing が所有する全バッファ（slot + **通常入力** — 破棄と readback 適格判定の分岐に使う）。
   * MUST: 常駐入力の実体は含めない — 所有者は GpuContext 側で、寿命は Session を跨ぐ。
   */
  readonly owned: ReadonlySet<GPUBuffer>;
  /**
   * readback を許す唯一の集合 = pin された slot + 入力バッファ。
   * MUST: 入力を含める — グラフ出力が入力の別名になる形（reshape(入力)）では読み戻し先が
   * 入力バッファそのものになる。非 backed 経路の入力はプール外（`allocHostWritten`）で
   * `RunArena.isReadable` が素通しにしていたので、これはその同値の再現。
   */
  readonly readable: ReadonlySet<GPUBuffer>;
};

/**
 * 読み戻し 1 本ぶんの staging と、ホストの {@link Tensor} へ組み直すのに要る宣言情報。
 * 積む先（run 本体のコマンド列 / readback 専用 encoder）に依らず同じ形で持つ。
 */
type StagedOutput = {
  readonly name: string;
  readonly shape: readonly number[];
  readonly count: number;
  /** `COPY_DST | MAP_READ`（{@link RunArena.allocHostRead}）。所有はアリーナ。 */
  readonly staging: GPUBuffer;
};

/** 検査済みの `copyOutputs` 1 件（{@link Session.#planCopyOutputs} — 実体はまだ解決していない）。 */
type PlannedCopy = {
  readonly name: string;
  readonly target: ResidentTensor;
  readonly size: number;
};

/** 写し元まで解決した `copyOutputs` 1 件（{@link Session.#resolveCopyOutputs}）。 */
type ResolvedCopy = {
  readonly source: GPUBuffer;
  readonly target: GPUBuffer;
  readonly size: number;
};

/**
 * 融合 attention の整数内積変種を決める（{@link SessionState.attentionI8a8Dot} の入口）。
 *
 * カナリア（src/gpu/attention-dp4a-canary.ts）を走らせるのは「拡張を広告していて、かつ a8 を
 * 要求された」ときだけ:
 *
 * - `I8A8_DOT` 指定時は走らせない — テストが変種を強制している最中に環境判定を挟むと、
 *   何を測ったのかが診断からも数値からも消える。
 * - 非広告 → dp4a 変種は生成すらされないので判定する対象が無い（従来どおり emu 直行）。
 * - a8 以外 → i8a8 の attention カーネルが 1 本も出ないので、この席の値は 1 度も読まれない。
 *   「使わない機能の初回コスト」を全 Session に配らないための門で、判定は最初に a8 を要求した
 *   Session が払い、以後は device 単位でメモ化される。
 *
 * カナリアが「既知解と厳密一致ではないが sanity 帯には収まった」で決めた場合は**黙って
 * 通さない** — 警告をメモの実体の中で出すことで、device 単位に 1 度だけになる（Session ごとに
 * 出すと a8 の Session を並べただけで同じ 1 事実が繰り返し流れる）。
 */
const resolveAttentionI8a8Dot = async (
  gpu: GpuContext,
  forced: I8a8Dot | undefined,
  attentionCompute: ComputePrecision,
  dp4a: boolean,
): Promise<I8a8Dot> => {
  if (forced !== undefined) return forced;
  if (!dp4a) return "emu";
  if (attentionCompute !== "a8") return "dp4a";
  const decision = await gpu[RUNTIME_INTERNAL].attentionI8a8Dot(async () => {
    const decided = await decideAttentionI8a8Dot(gpu);
    if (!decided.exact) warnInexactAttentionCanary(decided);
    return decided;
  });
  return decision.dot;
};

/**
 * カナリアが厳密一致を得られないまま帯内で決めたことを 1 回だけ知らせる。
 *
 * 止めないのは、この形が実在の健全な device（Apple M2 — 共有 f32 エピローグの丸めが既知解と
 * 数 ULP ずれるだけ）だからで、黙らないのは「帯内だから通した」が**測定に効く事実**だから
 * （a8 の出力はこの device で他機とビット同一にならない）。文言は @karume/hub の main 追従警告
 * と同じ流儀 — 何が起きたか・何をすれば消えるかを 1 本の console.warn で出す。
 */
const warnInexactAttentionCanary = (decision: AttentionI8a8Decision): void => {
  console.warn(
    `@karume/runtime: 融合 attention の i8a8 カナリアが既知解と厳密一致しなかった。\n` +
      `  ${formatAttentionI8a8Decision(decision)}\n` +
      `この device の a8 attention は他機とビット同一にはならない（差は sanity 帯の内側で、\n` +
      `共有 f32 エピローグの丸め差の水準）。ビット同一が要るなら attentionCompute を 'f32' か\n` +
      `'f16' にすること。`,
  );
};

type SessionState = {
  readonly gpu: GpuContext;
  /**
   * 実行するグラフ。MUST: `KarumeModel`（graph + file）を丸ごと持たない — file を掴むと
   * 配布ファイル全量の ArrayBuffer が Session の寿命まで固定され、shard 逐次消費
   * （ADR 0070 決定 3）の「参照を手放す」契約が成立しない。構築後に要るのは graph だけ。
   */
  readonly graph: IrGraph;
  readonly cache: PipelineCache;
  readonly scheduler: SubmitScheduler;
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  /**
   * params バッファの内容アドレスキャッシュ（キー = usage + 全要素の連結 —
   * `RecipeBuilder.#writeParams`）。実体は weights アリーナが所有する Session 常駐バッファで、
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
   * 圧縮のまま常駐した重み（席と付随実体 — ADR 0018 / 0019 / 0069）。ここに無い値は f32 と
   * して読む — カーネル変種の選択も追加束縛もこの表 1 つで決まる。
   *
   * MUST: 席・scale・group 長を並列 Map に割らない（{@link ResidentWeight} の doc）。載せるのは
   * Session 構築の 1 箇所だけで、group 長は宣言（graph）から写す — 別の値を渡せる形にすると
   * 「group 64 の資産が group 32 のパイプラインで走る」沈黙誤値になる。
   */
  readonly residentWeights: ReadonlyMap<string, ResidentWeight>;
  readonly storage: StorageDiagnostics;
  /** 構築相の費用内訳（{@link SessionBuildStats}）。構築の決着で確定し、以後不変。 */
  readonly buildStats: SessionBuildStats;
  /** linear の実行形（opt-in — {@link SessionOptions.linearCompute}）。 */
  readonly linearCompute: "f32" | "a8" | "f16";
  /** 融合 attention の実行形（opt-in — {@link SessionOptions.attentionCompute}）。 */
  readonly attentionCompute: ComputePrecision;
  /** S の格納形（opt-in — {@link SessionOptions.attentionScoreStorage}）。計算形と直交する軸。 */
  readonly attentionScoreStorage: ScoreStorage;
  /**
   * **linear の** i8a8 整数内積変種。既定は `navigator.gpu.wgslLanguageFeatures` の列挙から
   * 決まり、テストは {@link I8A8_DOT} で強制できる。**どちらでも数値は 1 ビットも変わらない**
   * （linear は Metal を含めて実走で反証されていない — docs/known-issues.md）。
   */
  readonly linearI8a8Dot: I8a8Dot;
  /**
   * **融合 attention の** i8a8 整数内積変種（①QK / ③PV）。linear と席を分けてあるのは、
   * 「両変種はビット同一」が attention だけ実機で反証されている（Metal / Apple M2 —
   * docs/known-issues.md）ため。既定は device 単位の実走カナリア
   * （src/gpu/attention-dp4a-canary.ts）が決め、テストは {@link I8A8_DOT} で強制できる。
   */
  readonly attentionI8a8Dot: I8a8Dot;
  /** 行ブロック枚数の強制（テスト専用 — {@link ROW_BLOCK_SPLIT}）。 */
  readonly rowBlockSplit: number | undefined;
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
  /**
   * 導出相（ステップ列 → レシピ列 — {@link RecipeBuilder}）。Session が持つのは 1 個だけで、
   * 状態は自身の {@link Session.#state} をそのまま渡す（run 寿命の器は面に載らない）。
   */
  readonly #recipeBuilder: RecipeBuilder;
  /** 活性 signature の slot backing（容量 1 — {@link ActiveBacking}）。 */
  #backing: ActiveBacking | undefined;
  #backingBuilds = 0;
  /**
   * 破棄待ちの slot バッファ。切替・追い出し・dispose はここへ積むだけで、実際の `destroy()` は
   * **flush 後の 1 箇所**（{@link Session.#destroyRetired}）でだけ起きる。
   */
  readonly #retired: GPUBuffer[] = [];
  /**
   * 生存中の {@link GenerationContext}（診断 `stateBacking.residentBytes` の**導出元**）。
   *
   * MUST: 常駐バイト数を独立に更新するカウンタで持たない — 生成と dispose の 2 経路が別々に
   * 足し引きする形にすると、片方を通らない失敗経路（確保途中の例外）で恒久的にずれる。
   */
  readonly #contexts = new Set<GenerationContext>();
  #contextCount = 0;
  /**
   * state を含む bind group を焼き直した累計回数（診断 `stateBacking.rebindCount`）。
   *
   * MUST: Session 累計で数える（context ごとに分けない）。観測したいのは「context を交互に
   * 使うと切替のたびに焼き直しが走る」形そのもので、それは Session の run 列に対する回数で
   * しか見えない。
   */
  #stateRebinds = 0;
  /**
   * 実行中 / 待機中の run と dispose の直列化チェーン。決着（成功・失敗）だけを次に渡すため
   * 自身は決して reject しない。
   */
  #chain: Promise<void> = Promise.resolve();
  /**
   * 発行済みで未決着の {@link Session.run} の本数（batch の自己デッドロック検出器の観測点）。
   *
   * MUST: 増減はどちらも**同期区間**で行う（`run` の呼び出し区間で +1・本体の finally で −1）。
   * 呼び出し順 = `#chain` の順なので、`enqueue` の同期区間でこれが正なら「未決着の run が
   * enqueue より前に居る」= 閉路の成立条件そのものになる（{@link Session.enqueue}）。
   * MUST: 数えるのは `run` だけ。errorScope 区間ロックを要求するのは run 本体ただ 1 つで、
   * dispose 系（`Session.dispose` / `GenerationContext.dispose`）は flush しか待たないため
   * 閉路に参加できない（それらを混ぜると、決着する列まで拒否する過剰な門になる）。
   */
  #pendingRuns = 0;
  #disposal: Promise<void> | undefined;

  private constructor(state: SessionState) {
    this.#state = state;
    this.#recipeBuilder = new RecipeBuilder(state);
  }

  /**
   * 構築の共通経路（全量面 = グラフ shard 1 本の列 / shard 面 = グラフ shard + N 重み shard）。
   * shard ごとに「進行検証 → errorScope 同期区間でアップロード → 明示 submit + フェンス」を
   * 刻み、全 shard 読了後に宣言完全性を検査する（ADR 0070 決定 3・4）。
   *
   * MUST: 構築の入口は {@link PreparedModel.createSession} ただ 1 つ（重みアップロードを含む
   * 明示 async ステージ）。クラス外から呼べる形なのは PreparedModel が別クラスだからで、
   * 公開面ではない — mod.ts は Session を型としてのみ出す。
   * MUST: 常駐計画（席）は prepare 相が求めた 1 個を受け取る — 構築側で引き直すと、見積りが
   * 見た席と実際に上げる席が別の計算結果になりうる形が戻る。
   */
  static async build(
    gpu: GpuContext,
    graph: IrGraph,
    residency: ReadonlyMap<string, WeightResidency>,
    shards: AsyncIterable<WeightShard>,
    options: SessionOptions,
  ): Promise<Session> {
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
          "shader-f16 無しで S を半分にするなら attentionCompute を 'f32' か 'a8' にすること）",
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

    // 整数内積変種は **linear と attention で別席**（{@link SessionState}）。どちらも
    // `I8A8_DOT` の指定が最優先で、指定が無ければ族ごとの既定に落ちる。
    const dp4a = dp4aAvailable(gpu.wgslLanguageFeatures);
    const attentionI8a8Dot = await resolveAttentionI8a8Dot(
      gpu,
      options[I8A8_DOT],
      attentionCompute,
      dp4a,
    );

    const scheduler = new SubmitScheduler(gpu, options.submitPolicy);
    const weights = new RunArena(gpu.device, () => scheduler.flush());
    const weightBuffers = new Map<string, GPUBuffer>();
    const residentWeights = new Map<string, ResidentWeight>();
    let residentCompressedBytes = 0;
    let hostExpandedBytes = 0;
    // 構築相の費用内訳（{@link SessionBuildStats}）。ホスト時計だけで刻む集計器で、
    // MUST NOT: 計測のために GPU フェンスを足さない・submit の位置を動かさない
    // （shard ごと submit 1 回という ADR 0070 決定 3 の契約が崩れると、瞬間ピークが重み 1 本ぶん
    // 押し上がる）。よって writeBuffer の実転送時間は uploadFenceMs に吸われたままになる。
    let shardCount = 0;
    let shardWaitMs = 0;
    let decodeMs = 0;
    let bufferCreateMs = 0;
    let writeBufferIssueMs = 0;
    let uploadedBytes = 0;
    let uploadFenceMs = 0;
    // 計測の巻き付けは 3 経路（decode / createBuffer / writeBuffer）とも呼び出し点が複数あるので
    // 局所ヘルパに畳む。**呼び出しの順序も引数も 1 つも変えない**（計測は素通しの薄い層）。
    const timedDecode = (decode: () => Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
      const start = performance.now();
      const expanded = decode();
      decodeMs += performance.now() - start;
      return expanded;
    };
    const timedAlloc = (bytes: number): GPUBuffer => {
      const start = performance.now();
      const buffer = weights.allocHostWritten(bytes, HOST_WRITTEN_USAGE);
      bufferCreateMs += performance.now() - start;
      return buffer;
    };
    const timedWrite = (
      buffer: GPUBuffer,
      data: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    ): void => {
      const start = performance.now();
      gpu.device.queue.writeBuffer(buffer, 0, data);
      writeBufferIssueMs += performance.now() - start;
      uploadedBytes += data.byteLength;
    };
    // 宣言と実テンソルの突合・完全性は shard 進行検証に一本化（ADR 0070 決定 1 — 全量面も
    // 同じ門を通る。openModel 済みの入力には冪等）。
    const validator = createShardValidator(graph);
    try {
      // shard の反復待ち（= 供給側の費用）は for await が隠すので、**前の shard を処理し終えた
      // 時刻**との差で測る（次の shard が届くまでの間はこの 2 点の間にしか無い）。
      let shardBoundary = performance.now();
      for await (const shard of shards) {
        shardWaitMs += performance.now() - shardBoundary;
        shardCount += 1;
        // errorScope とフェンスは同じラベルを名乗る MUST（別々に組むと同じアップロード区間の
        // 失敗が 2 つの名前で出る）。
        const label = uploadLabel(shard.origin);
        let ready: readonly ReadyInitializer[];
        try {
          ready = validator.intake(shard.file);
        } catch (cause) {
          // 宣言違反・co-shard・余剰・shard 横断重複はその shard の中身を直す話なので、
          // 帰属先はファイル名（全量面は素通し = 従来文言）。
          throw attributeToShard(shard.origin, cause);
        }
        // MUST: 重みアップロードも errorScope で囲む（ADR 0004 の「errorScope 常設」）。上限超過の
        // createBuffer は同期例外を投げずに無効バッファを返し、無効バッファ / 整列違反への
        // writeBuffer も警告すら出さない no-op になるため、包まないと重みが空のまま走り出す。
        // MUST NOT: この区間の中で await しない。push から pop の発行までを 1 つの同期区間に
        // 保つことが、device 単位ロックを取らずに LIFO の交錯を防いでいる根拠になっている。
        // 区間は shard 単位（ADR 0070 決定 4 — 網の撤去ではなく粒度の変更。次 shard の取得と
        // フェンスの await は区間の外に出る。副次利得として失敗 shard の特定が細かくなる）。
        pushFailureScopes(gpu.device);
        try {
          for (const item of ready) {
            const name = item.name;
            const initializer = graph.initializers[name];
            const raw = tensorBytes(item.file, item.view);
            // 席はプランナが正本（全 initializer を載せる契約 — 欠けは簿記の破れ）。
            const seat = residency.get(name);
            if (seat === undefined) {
              throw new ExecutionError(`initializer '${name}': 常駐分類が無い`);
            }
            // MUST: 宣言由来のバイト長と現物が食い違ったら落とす。プランナ（と見積り）は実
            // テンソルを見ずに宣言だけで数えるので、ここが「宣言 = 現物」を実際に確かめる唯一の
            // 点になる（container の突合門が成立していれば発火しない — 二重の網）。
            if (raw.byteLength !== seat.payloadBytes) {
              throw new ExecutionError(
                `initializer '${name}': 宣言由来 ${seat.payloadBytes} バイトに対し実テンソルが ${raw.byteLength} バイト`,
              );
            }
            // 格納 f16 / i8 / i4 だけが 2 経路に分かれる（ADR 0018 / 0019 / 0069）。適格なら
            // 生バイトのまま常駐させ dequant はカーネル内（VRAM 削減はこれで初めて成立する）、
            // 適格外はここで f32 へ展開する（正しさは保たれ VRAM 削減はゼロ）。他の格納 dtype は
            // 生バイトがそのまま GPU 表現。
            let payload: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer> = raw;
            if (initializer.storage.dtype === "f16") {
              if (seat.seat === "f16") {
                // MUST: 奇数要素長は末尾 2 バイトのゼロ詰めで 4 バイト整列させる。writeBuffer は
                // 4 の倍数でないサイズを validation で拒む（= 重みが空のまま走り出す）。
                payload = alignF16Payload(raw);
                residentWeights.set(name, { storage: "f16" });
                residentCompressedBytes += payload.byteLength;
              } else {
                payload = timedDecode(() => decodeF16(raw));
                hostExpandedBytes += payload.byteLength;
              }
            }
            if (initializer.storage.dtype === "i8") {
              const scale = scaleTensor(item, "i8");
              // initializer の宣言 shape は数値のみ（parseIrGraph が保証 — 記号次元は拒否）。
              const shape = graph.values[name].shape.map(Number);
              if (seat.seat === "i8") {
                assertChannelScale(name, shape, scale.shape, seat.channelAxis);
                // MUST: 要素数が 4 の倍数でない重みは末尾をゼロ詰めして 4 バイト整列させる
                // （f16 の 2 バイト詰めと同じ理由 — writeBuffer が validation で落ちる）。
                payload = alignI8Payload(raw);
                // MUST: scale のバッファも「GPU 常駐圧縮」に数える（実際に抱えるバイト数）。
                residentCompressedBytes += payload.byteLength + scale.bytes.byteLength;
                const scaleBuffer = timedAlloc(Math.max(4, scale.bytes.byteLength));
                if (scale.bytes.byteLength > 0) {
                  timedWrite(scaleBuffer, scale.bytes);
                }
                residentWeights.set(name, { storage: "i8", scale: scaleBuffer });
              } else {
                payload = timedDecode(() => decodeI8(raw, shape, scale.values, scale.shape));
                hostExpandedBytes += payload.byteLength;
              }
            }
            if (initializer.storage.dtype === "i4") {
              const scale = scaleTensor(item, "i4");
              const shape = graph.values[name].shape.map(Number);
              // 適格は f16 / i8 より狭い「消費が linear / embedding / conv1d(groups==1) の
              // 重みスロットのみ」（ADR 0069 決定 5 とその追補 — 展開経路が GEMM 骨格のタイル
              // 読み〈linear は B 側・conv1d igemm は A 側〉と embedding のカーネルにしか無い）。
              // 展開経路の無い重みスロット（conv2d / conv_transpose1d / groups > 1 の conv1d）と
              // 共有される i4 は CPU 展開の受け皿へ（正しさは保たれ VRAM 削減はゼロ —
              // i8 の適格外と同じ設計）。判定はプランナが済ませている。
              if (seat.seat === "i4") {
                // ペイロードは詰め物不要で常に 4 バイト整列 — バイト長 = numel / 2 で、numel は
                // group_size（2 冪 ≥ 16）の倍数だからバイト長は 8 の倍数（ADR 0069 決定 2）。
                // MUST: scale のバッファも「GPU 常駐圧縮」に数える（i8 と同じ — 実際に抱える
                // バイト数。exporter の storage_breakdown と診断の意味を揃える）。
                residentCompressedBytes += payload.byteLength + scale.bytes.byteLength;
                const scaleBuffer = timedAlloc(Math.max(4, scale.bytes.byteLength));
                if (scale.bytes.byteLength > 0) {
                  timedWrite(scaleBuffer, scale.bytes);
                }
                // group 長は宣言から写した 1 箇所（プランナ）だけが決める — 別経路で渡せる形に
                // すると「group 64 の資産が group 32 のパイプラインで走る」沈黙誤値になる。
                residentWeights.set(name, {
                  storage: "i4",
                  scale: scaleBuffer,
                  groupSize: seat.groupSize,
                });
              } else {
                // 値域（2 冪 ≥ 16・整除）は parseIrGraph が保証済み。存在は型の上でだけ optional
                // なので、黙って読み飛ばさず言い直す（「格納 i8 なのに scale が無い」と同じ流儀）。
                const groupSize = initializer.storage.groupSize;
                if (groupSize === undefined) {
                  throw new ExecutionError(
                    `initializer '${name}': 格納 i4 なのに group_size が無い`,
                  );
                }
                payload = timedDecode(() =>
                  decodeI4(raw, shape, scale.values, scale.shape, groupSize)
                );
                hostExpandedBytes += payload.byteLength;
              }
            }
            const buffer = timedAlloc(Math.max(4, payload.byteLength));
            if (payload.byteLength > 0) timedWrite(buffer, payload);
            weightBuffers.set(name, buffer);
          }
        } catch (cause) {
          // MUST: push した 2 本は必ず pop して積み残さない（積み残すと以後の検証結果が誤った
          // スコープに吸われ、エラーが恒久的に見えなくなる）。破棄は外側の transaction 境界が
          // 1 箇所で持つ。
          await discardFailureScopes(gpu.device);
          throw attributeToShard(shard.origin, cause);
        }
        const failure = await popFailureScopes(gpu.device, label);
        if (failure !== undefined) throw failure;

        // MUST: shard ごとに**実際の submit を 1 回**出して完了まで待つ（ADR 0070 決定 3）。
        // queue.writeBuffer は staging を確保して溜め込み、submit の完了までそれを解放しない —
        // 数 GiB の重みを上げた直後は VRAM が二重計上のまま最初の run に入り、初回ピークが
        // 重み 1 本ぶん押し上がる（f16 preset で実測 +2.7GiB。
        // docs/research/2026-08-08-vram-oom-misreport.md §4）。shard 逐次消費ではこの解放が
        // RAM ピーク O(最大 shard) の成立条件そのものになる。フェンスの後にループ末尾へ抜けて
        // shard.file への参照が尽きる — CPU 側バイト列は転送完了後にだけ手放される
        // （フェンス後解放の順序契約 — ADR 0070 決定 3）。
        // MUST NOT: scheduler.flush() で代用しない。pending dispatch が空だと submit を出さずに
        // 即 return するため、staging は溜まったまま残る。
        // NOTE: submit ごとの onSubmittedWorkDone を禁じているのは run のホットパス（submit.ts の
        // 「計測の帰属」）で、ここは shard ごと 1 回・窓の外なので推定にも壁時計にも乗らない。
        // NOTE: errorScope で囲まないのは、空の submit が確保も検証も伴わないため（両建てで囲む
        // のは「確保を伴う区間」— device.ts の pushFailureScopes）。加えて Session の構築は
        // GpuContext のスコープロック外なので、await を跨ぐスコープをここに張ると並行 Session の
        // 失敗を誤帰属させる口になる。
        gpu.device.queue.submit([]);
        // MUST: 消失後の onSubmittedWorkDone が解決しない実装がありうる（実測は
        // raceCanaryDeviceLost の doc）ため競わせる — ハングを失敗に変換する保険。
        const fenceStart = performance.now();
        await gpu[RUNTIME_INTERNAL].raceDeviceLost(
          gpu.device.queue.onSubmittedWorkDone(),
          label,
        );
        uploadFenceMs += performance.now() - fenceStart;
        shardBoundary = performance.now();
      }
      // 宣言完全性（欠け）は全 shard を読み終えて初めて判定できる（ADR 0070 決定 1）。
      validator.finish();
    } catch (cause) {
      // transaction 境界（ADR 0070 決定 3）: 途中の shard で失敗したら（宣言違反・入力列の例外・
      // GPU エラーのいずれでも）、アップロード済みの重みごと weights アリーナを破棄して
      // 部分 Session を公開しない。
      // MUST: 後始末の失敗で本体の例外を上書きしない（run 側と同じ規律）。原因は本体側に
      // あり、destroy の rejection（主因は device 消失）に差し替わると調査の起点が消える。
      await weights.destroy().catch(() => undefined);
      throw cause;
    }

    return new Session({
      gpu,
      graph,
      cache: new PipelineCache(gpu.device),
      scheduler,
      weights,
      weightBuffers,
      paramsCache: new Map(),
      prepared: new Map(),
      residentWeights,
      storage: { residentCompressedBytes, hostExpandedBytes },
      buildStats: {
        shardCount,
        shardWaitMs,
        decodeMs,
        bufferCreateMs,
        writeBufferIssueMs,
        uploadedBytes,
        uploadFenceMs,
      },
      linearCompute,
      attentionCompute,
      attentionScoreStorage,
      // linear の拡張の有無は**速度にしか効かない**（両変種は同じ整数を返す）ので、機能検出では
      // なく経路選択としてここで 1 度だけ決める（src/kernels/linear-i8a8.ts の docstring）。
      linearI8a8Dot: options[I8A8_DOT] ?? (dp4a ? "dp4a" : "emu"),
      // attention は同じ主張が実機で反証されている（Metal / Apple M2）ので、列挙ではなく
      // **実走カナリアの判定**（上の `attentionI8a8Dot`）で決める。
      attentionI8a8Dot,
      rowBlockSplit: options[ROW_BLOCK_SPLIT],
      useCounts: countUses(graph),
      dtypes: declaredDtypes(graph),
      outputNames: new Set(graph.outputs),
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
   *
   * `generation` を渡した run は **state 参照グラフの 1 step**（ADR 0066 決定 4 の prefill-chunk
   * または decode）になる。渡さない run は 1 バイトも挙動が変わらない。
   *
   * ## 入力の寿命（発行時 snapshot と borrowed な data）
   *
   * 実行本体は `#serialize` 越し（1 マイクロタスク以降）に走るので、「非 await の並行発行」と
   * 「発行直後の書き換え」の組は素直に踏める形になっている。そこで **metadata は発行の同期区間で
   * 固定する**（{@link captureInputs}）— `inputs` Record の member 構成・各入力の shape・
   * `bindings` は写しを取るので、戻り Promise を待たずに呼び出し側が書き換えても、この run は
   * **発行時点の形**で走る。
   *
   * MUST NOT: `Tensor.data`（TypedArray の実体）は写さず**借りる**ので、戻り Promise が settle
   * するまで書き換えない。GPU への `writeBuffer` はマイクロタスクの先で出るため、書き換えは
   * 例外も警告も無い沈黙誤値になる。GiB 級の複製を毎 run 払わないための意図的な線引きで、
   * 「借りる」ことそのものが契約（常駐入力 {@link ResidentTensor} が束縛予約と破棄済み検査で
   * 守られているのと同じ規律を、ホスト配列では呼び出し側が守る）。
   */
  run(
    inputs: RunInputs,
    bindings: SymbolBindings = {},
    generation?: GenerationRun,
  ): Promise<RunOutputs> {
    // dispose 済みの判定は呼び出し時点で行う（チェーンの中で見ると、dispose より前に発行した
    // run まで巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      return Promise.reject(new ExecutionError("dispose 済みの Session では実行できない"));
    }
    // MUST: context のリース取得は `#serialize` に積む**前**（`enqueue` の `batch.enter` と
    // 同じ理由 — 本体はマイクロタスクを 1 段挟むので、本体で取ると「未 await の run の直後に
    // `context.rewind()`」が「進行中 run が居ない」と判定され、捕捉済み P と uniform が分裂する）。
    // 検査の失敗は従来どおり戻り Promise の reject で返す（同期 throw に変えない）。
    const lease = generation?.context[RUNTIME_INTERNAL];
    let captured: CapturedInputs;
    let capturedBindings: SymbolBindings;
    try {
      // MUST: 入力の写しはリース取得より**前**（写しが落ちた後に返し手の居ないリースが 1 本
      // 残ると、以後の `rewind()` が永久に拒否される）。
      captured = captureInputs(inputs);
      // 束縛も発行時に固定する（本体で読むと、発行直後の書き換えが「シンボルの束縛が衝突」
      // という無関係な失敗に化ける）。
      capturedBindings = { ...bindings };
      lease?.acquireRun();
    } catch (cause) {
      return Promise.reject(cause);
    }
    // MUST: 同期区間で数える（本体はマイクロタスクを 1 段挟むので、本体で数えると同じ tick に
    // 積まれた `enqueue` から「未決着の run」が見えない）。
    this.#pendingRuns += 1;
    return this.#serialize(async () => {
      try {
        return await this.#runOnce(captured, capturedBindings, generation);
      } finally {
        // MUST: 成功・失敗のどちらでも必ず返す（返し損ねると以後の rewind が永久に拒否される）。
        lease?.releaseRun();
        this.#pendingRuns -= 1;
      }
    });
  }

  /**
   * **フェンスを張らずに** 1 回ぶんの実行を積む（H-5 — 生成ループの run 境界を消す面）。
   *
   * `run` との違いは 3 つだけ: ①`flush` / `onSubmittedWorkDone` / readback を出さない
   * （待ちは {@link BatchScope.finish} の 1 本に集約される）②出力はホストへ返らず、
   * `copyOutputs` で常駐テンソルへ書き出す ③errorScope は batch が張ったものに相乗りする
   * （run ごとの push / pop は出ない）。積むコマンド列そのものは backed run と**同一**。
   *
   * MUST: 通る経路は「導出済み計画 + slot backing」だけ。初回は導出と backing 構築をここで
   * 済ませる（run と違って**初回から** backing を作る — 作らなければ次も非 backed になり、
   * enqueue が成立しない）。アリーナ経路・readback 経路へ**黙って退避しない**のがこの面の
   * 前提で、退避が要る形（出力をホストで受けたい等）は `run` を使うこと。
   * MUST: 末尾で必ず eager submit する（{@link SubmitScheduler.submitPending}）。これが
   * 「次の enqueue / `writeBuffer` が先行 dispatch を追い越さない」の根拠。
   * MUST: 同一 Session の run / enqueue / dispose は呼び出し順に直列化される（`run` と同じ
   * {@link Session.#chain}）。
   * MUST: **未決着の `run` を先に持つ Session からは enqueue できない**（fail loudly）。
   * その列は確定的な自己デッドロックになる: batch 区間は errorScope 区間ロックを保持したまま
   * in-flight リースの返却を待ち → リースは enqueue 本体の finally でしか返らず → enqueue 本体は
   * `#chain` の先行 run の決着を待ち → run 本体はその区間ロックを待つ、で 4 辺が閉じる。
   * `withScopeLock` は「正当な待ち行列」と「再入」を区別できず再入検出器を置けない設計なので、
   * 検出しなければ**例外も診断も出ないまま永久にハングする** — ハングを型付き例外へ変換する。
   * NOTE: 戻り Promise を await せずに {@link BatchScope.finish} を呼んでも取りこぼさない。
   * batch の in-flight リースをこの**同期区間**で取り、finish は未返却リースが全て返るまで
   * フェンスへ進まない（機構は `BatchInternals.enter`）。
   *
   * 入力の寿命は `run` と**同型**（{@link Session.run} の「入力の寿命」節）— `inputs` Record の
   * member 構成・shape・`bindings` / `copyOutputs` の形は発行の同期区間で固定し、`Tensor.data`
   * は戻り Promise が settle するまで borrowed（MUST NOT: 書き換えない）。非 await 発行が
   * 前提の面なので、踏みやすさは run より高い。
   */
  enqueue(inputs: RunInputs, options: EnqueueOptions): Promise<void> {
    if (this.#disposal !== undefined) {
      return Promise.reject(new ExecutionError("dispose 済みの Session では実行できない"));
    }
    // MUST: リースを取る**前**に見る（取ってから落とすと、返し手の居ないリースが 1 本残って
    // `finish()` が今度こそ永久に待つ）。
    if (this.#pendingRuns > 0) {
      return Promise.reject(
        new BatchScopeError(
          `未決着の run が ${this.#pendingRuns} 本ある Session には enqueue できない` +
            "（batch 区間は errorScope 区間ロックを握ったまま enqueue の決着を待ち、その " +
            "enqueue は先行 run を待ち、run はそのロックを待つ = 自己デッドロック）。" +
            "run を await してから batch を開くか、区間中は enqueue だけを使うこと",
        ),
      );
    }
    const batch = options.batch[RUNTIME_INTERNAL];
    // MUST: 受け口の検査とリース取得は `#serialize` に積む**前**に済ませる。本体はマイクロ
    // タスクを 1 段挟むので、本体で取ると「未 await の enqueue を積んだ直後に finish()」で
    // finish が先に決着し、積んだぶんが 1 本も dispatch されないまま区間が成功で終わる。
    // 検査の失敗は従来どおり戻り Promise の reject で返す（同期 throw に変えない）。
    let captured: CapturedInputs;
    let capturedOptions: EnqueueOptions;
    try {
      // MUST: 入力の写しはリース取得より**前**（写しが落ちた後に返し手の居ないリースが 1 本
      // 残ると `finish()` が永久に待つ）。`batch` は実体そのものを持つ（写す対象ではない）。
      captured = captureInputs(inputs);
      capturedOptions = {
        batch: options.batch,
        bindings: { ...options.bindings },
        ...(options.copyOutputs === undefined
          ? {}
          // 写し先の Record も member 差し替えを固定する（入力と同じ理由 — 差し替えが通ると
          // 「書けたのに別の常駐テンソルへ」の沈黙誤値になる）。実体は借りたまま。
          : { copyOutputs: { ...options.copyOutputs } }),
      };
      batch.enter(this.#state.gpu);
    } catch (cause) {
      return Promise.reject(cause);
    }
    return this.#serialize(async () => {
      try {
        await this.#enqueueOnce(captured, capturedOptions);
      } finally {
        // MUST: 成功・失敗のどちらでも必ず返す（返し損ねると finish がハングする）。
        batch.leave();
      }
    });
  }

  /**
   * 1 生成ぶんの可変 state を所有する {@link GenerationContext} を作る（ADR 0066 決定 1 / 6）。
   *
   * `graph.states` のスロット容量（記号次元は `spec.bindings` で解決）と `spec.chunkLength` を
   * 確定して物理確保する。Session 側（不変重み・計画キャッシュ・slot backing）は**何も変わらない**
   * — 決定 5 の所有権分離により、レシピと計画鍵は context を知らない。
   *
   * MUST: state スロットは Session のアリーナにも {@link ResidentTensor} にも載せない（寿命の
   * 粒度が違う — 詳細は generation-context.ts のモジュール doc）。
   * MUST: 確保は out-of-memory / validation の errorScope 区間で行い、失敗は fail loudly
   * （決定 6）。容量が `maxStorageBufferBindingSize` を超えるスロットは確保の前に落とす（追記 5）。
   * NOTE: context は Session より長生きできる（`dispose` は注入した flush と自分のバッファだけを
   * 触る）。ただし state を使う run は Session 経由なので、実際の用途は Session の生存中に閉じる。
   * NOTE: prefill / decode の呼び出し形（ADR 0066 決定 6 が実装設計へ委ねた部分）は
   * **`Session.run` の第 3 引数 1 本**に落ちた（{@link GenerationRun}）。2 つの実行形を分ける
   * のは `queryLength` と入力の物理 chunk 行数だけで、メソッドは増やさない。
   */
  createGenerationContext(spec: GenerationContextSpec): Promise<GenerationContext> {
    // dispose 済みの判定は呼び出し時点で行う（run / enqueue と同じ規律）。
    if (this.#disposal !== undefined) {
      return Promise.reject(
        new ExecutionError("dispose 済みの Session では GenerationContext を作れない"),
      );
    }
    return this.#createGenerationContext(spec);
  }

  /**
   * {@link Session.createGenerationContext} の本体（借りる面の組み立てと生存集合の登録）。
   *
   * MUST: 生存集合への登録は**確保が決着した後**（失敗した context を数えると、診断の常駐
   * バイト数が実体の無いぶんを主張し続ける）。
   * MUST: dispose 判定は確保の**後にもう一度**執行する。`#disposal` は `dispose()` の同期区間で
   * 立つので、`GenerationContext.create` の await を跨いで dispose が発行されると呼び出し時点の
   * 判定をすり抜け、dispose 済み Session の生存集合へ登録されてしまう（診断
   * `stateBacking.residentBytes` が非ゼロを主張し続け、その GPU バッファは利用者が
   * `context.dispose()` を呼ぶまで残る）。確保済みの context は自分で破棄してから落とす。
   * NOTE: create 全体を {@link Session.#serialize} へ積む形は**採らない** — `#chain` の一員に
   * なると、batch 区間の未決着 run と同型の循環（BatchScope → enqueue → run → errorScope 区間
   * ロック）に create まで巻き込まれる。チェーン外であることが待ち合わせグラフ上は安全側。
   */
  async #createGenerationContext(spec: GenerationContextSpec): Promise<GenerationContext> {
    const host: GenerationContextHost = {
      gpu: this.#state.gpu,
      graph: this.#state.graph,
      flush: () => this.#state.scheduler.flush(),
      serialize: <T>(body: () => Promise<T>): Promise<T> => this.#serialize(body),
      forget: (context: GenerationContext): void => {
        this.#contexts.delete(context);
      },
    };
    const context = await GenerationContext.create(host, spec);
    if (this.#disposal !== undefined) {
      // 確保済みぶんを先に返してから fail loudly（黙って生存集合の外へ漏らさない）。
      await context.dispose();
      throw new ExecutionError("dispose 済みの Session では GenerationContext を作れない");
    }
    this.#contexts.add(context);
    this.#contextCount += 1;
    return context;
  }

  /** 重みバッファを解放する。実行中の run の完了を待ってから破棄し、以後の run は fail loudly。 */
  dispose(): Promise<void> {
    // MUST: 2 度目以降も同じ完了を返す。先に返すと呼び出し側が「破棄済み」と見なして
    // device.destroy() まで進み、flush-before-destroy が崩れる。
    // MUST: slot backing の破棄も**この 1 本に相乗り**させる（破棄経路の担い手を増やさない —
    // ADR 0004）。weights.destroy() は flush の完了まで待つので、その後の destroy は
    // flush-before-destroy を満たす。失敗（主因は device 消失）しても破棄は必ず行う。
    this.#disposal ??= this.#serialize(async () => {
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
      buildStats: this.#state.buildStats,
      lastRun: this.#lastRun,
      lastRunTiming: this.#state.scheduler.timing,
      lastRunFusions: this.#lastRunFusions,
      lastRunParams: this.#lastRunParams,
      lastRunPrepared: this.#lastRunPrepared,
      planBacking: {
        residentBytes: this.#backing?.bytes ?? 0,
        buildCount: this.#backingBuilds,
      },
      stateBacking: {
        // MUST: 生存集合から毎回導出する（{@link Session.#contexts} の doc）。
        residentBytes: [...this.#contexts]
          .reduce((total, context) => total + context[RUNTIME_INTERNAL].bytes, 0),
        contextCount: this.#contextCount,
        rebindCount: this.#stateRebinds,
      },
    };
  }

  /** 直前の実行の決着後に `body` を走らせる。戻り値はこの呼び出し自身の結果 / 例外。 */
  #serialize<T>(body: () => Promise<T>): Promise<T> {
    const result = this.#chain.then(body);
    this.#chain = result.then(() => undefined, () => undefined);
    return result;
  }

  /**
   * run 1 本の本体。**入力は発行時の写し**（{@link CapturedInputs}）で受け取る — ここは
   * 1 マイクロタスク以降に走るので、利用者の Record を引き直すと発行後の書き換えを読む。
   */
  async #runOnce(
    captured: CapturedInputs,
    bindings: SymbolBindings,
    generation: GenerationRun | undefined,
  ): Promise<RunOutputs> {
    const { gpu, graph, scheduler } = this.#state;

    // MUST: context の素性は**エンコードに入る前**に見る（この段の失敗は state に届かないので
    // poison しない）。別 Session / dispose 済みの取り違えは、通すと「別の生成の KV を束ねた
    // まま回る」沈黙誤値になり、値にしか出ない。
    if (generation !== undefined && !this.#contexts.has(generation.context)) {
      throw new ExecutionError(
        "run の GenerationContext がこの Session の生存集合に無い" +
          "（別の Session が作った context か、dispose 済み）",
      );
    }
    // 束縛解決済みのスロット容量。レシピと計画鍵に載るのは**この容量だけ**で、context の
    // 識別子は 1 バイトも載らない（ADR 0066 決定 5 — {@link Session.#preparedKey}）。
    const stateShapes = generation === undefined ? undefined : new Map(
      [...generation.context[RUNTIME_INTERNAL].slots].map(([name, slot]) => [name, slot.shape]),
    );
    // MUST: 論理長は run の頭で 1 度だけ読む（汚染・破棄・device 消失はこの読みが落とす）。
    // uniform へ書く値・dispatch 数の算出・容量の検査・進行が同じ 1 つの値から出ることが、
    // 「GPU が走査する範囲」と「ホストが撃った workgroup 数」の一致の根拠になる（下流の
    // `writeLengths` / `advance` はこの捕捉値を受け取り、context の現在値と照合する）。
    // MUST: 読むのは**内部面**（利用者面の `pastLength` ではない）。dispose は 2 段で、
    // 受理済みのこの run は 1 段目（受付終了）の後にも完走する契約のため。
    const pastLength = generation === undefined
      ? 0
      : generation.context[RUNTIME_INTERNAL].pastLength();
    /**
     * generation run の context 側の面。**context と encoding を 1 つの変数に束ねる**のは、
     * 焼き込み経路（{@link Session.#generationGroups}）が両方を同時に要るため — 別々に持つと
     * 「片方だけ undefined」という起こり得ない組を型で排除できず、握り潰しの分岐が生える。
     */
    const generationFace = generation === undefined ? undefined : {
      context: generation.context,
      encoding: generationEncoding(generation.context, pastLength, generation.queryLength),
    };
    const encoding = generationFace?.encoding;
    /**
     * 最初の state 書き dispatch を積んだ時点の submit カウンタ（undefined = まだ積んでいない）。
     * 失敗時の poison 判定（ADR 0066 追記 3）はこの値との比較 1 本で決まる。
     */
    let stateWriteSubmits: number | undefined;

    const { inputShapes, residentInputs } = captured;
    // MUST: 束縛の解決（= 入力 shape の検証）はヒット・ミスに関わらず**毎 run 走らせる**。
    // ここが飛ぶと、キャッシュに当たった run だけ入力 shape の宣言不一致を素通りする。
    const resolved = bindSymbols(graph, inputShapes, bindings, residentNames(residentInputs));
    if (generation !== undefined) {
      assertGenerationBindings(generation.context[RUNTIME_INTERNAL].bindings, resolved);
    }
    const preparedKey = this.#preparedKey(resolved, residentInputs, stateShapes);
    const prepared = this.#takePrepared(preparedKey);
    // 計画（planGraph）と融合判定（planFusions）はどちらも GPU に触れない純関数で、ヒット時は
    // 丸ごと飛ばす（根拠は {@link Session.#preparedKey}）。融合は掴めなかったノードを素のまま
    // ステップ列に並べるので、この段は「速くなるか」だけを決め、正しさには関与しない。
    const derived = prepared ?? this.#planSteps(resolved, stateShapes);
    const shapes = derived.shapes;
    // MUST: ヒット run も融合回数を報告する（ADR 0040 §3 の常設契約 — キャッシュの有無で
    // 観測点が消えると、融合が外れた状態がヒット run の裏に隠れる）。
    this.#lastRunFusions = derived.fusions;
    this.#lastRunPrepared = undefined;
    this.#recipeBuilder.resetParamsStats();

    const device = gpu.device;
    /**
     * この run のフェンスを **mapAsync の 1 本**に畳むか（H-1）。分岐はこの 1 箇所だけで、
     * 下流（アリーナの後始末・submit の出し方・読み戻しの積み先）は全てこの真偽で決まる。
     *
     * 二段待ちのまま据え置く条件は 2 つ:
     * - **gpuTiming が有効**: timestamp の回収も計測窓も `flush` の `onSubmittedWorkDone` に
     *   ぶら下がっている（ADR 0021 — 回収は完了後でなければ mapAsync が待ちに化ける）。計測は
     *   内訳を採る診断モードで、そこでフェンス 1 本を惜しむ理由が無い。
     * - **グラフ出力が 0 本**: 積む copy が無い = mapAsync が 1 本も出ない。無フェンスで返すと
     *   未完了の GPU 実行を残したまま run が戻るので、その run だけ従来の待ちへ落とす
     *   （IR は空の `graph.outputs` を受理する — format/ir.ts）。
     */
    const singleFence = !gpu.gpuTimingEnabled && graph.outputs.length > 0;
    // MUST: 単一フェンス経路の後始末は「未 submit のエンコードを残さない」までで、完了待ちは
    // 張らない（flush-before-destroy の完了ぶんは mapAsync が済ませており、submit 済み
    // コマンドから破棄済みバッファを参照するのは WebGPU 的に安全 — 実解放は完了まで実装が
    // 遅延する）。既定（二段待ち）は従来どおりフェンス付き flush のまま。
    const arena = new RunArena(
      device,
      singleFence
        ? () => {
          scheduler.submitPending();
          return Promise.resolve();
        }
        : () => scheduler.flush(),
    );
    // MUST: env は **run 寿命の実体だけ**（グラフ入力とノード出力）。Session 常駐の重み /
    // per-channel scale は導出相で直参照へ畳むので、run ごとに写す必要が無い — 写すと
    // 「run の器に Session 常駐の状態が混ざる」形が残り、レシピの再利用が成り立たなくなる。
    const env = new Map<string, GPUBuffer>();
    // この run が束縛予約を積んだ常駐入力（{@link Session.#bindInput}）。エンコードと submit が
    // 済んだ時点で必ず返す（成功経路・失敗経路の 2 箇所だけで、どちらか一方が必ず走る）。
    const boundResidents: ResidentTensor[] = [];
    // MUST: **run が GPU 操作を発行するのは自分のロック区間内のみ**（エンコード〜flush〜pop〜
    // readback〜アリーナ破棄まで）。errorScope は device 単位の LIFO で、操作の失敗は発行時点
    // のスタック先頭に帰属するため、「スコープを張らない操作だからロック外でよい」は成り立た
    // ない — 張らない操作の失敗こそ他人のスコープに入る。Session 内は #chain で直列化済みだ
    // が、同一 device 上の**別 Session** の run とは重なりうる（GpuContext の不変条件を参照）。
    // MUST NOT: この区間の内側からロックを再取得しない（自己デッドロック — 内側の層は同期
    // 区間で完結するスコープのみを使う）。
    const encode = async (): Promise<RunOutputs> =>
      await gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
        // GPU 時間内訳の寿命は **直近 run**（ADR 0021）。ロックを取ってから捨てることで、
        // 表が「今から積む run のぶんだけ」になる（先行 run の回収は既に済んでいる）。
        scheduler.resetTiming();
        let outputs: RunOutputs;
        // 活性化した slot backing（ミス run では undefined のまま）。readback の適格判定が
        // 「その実体が pin された slot か」を見るため、エンコード区間の外まで持ち越す。
        let backing: ActiveBacking | undefined;
        // この run が backing を新規構築したか（失敗時の回復規律 — 下の catch が読む）。
        let builtBacking = false;
        // 単一フェンス経路で run 本体のコマンド列へ積んだ読み戻し（undefined = 二段待ち経路）。
        let staged: readonly StagedOutput[] | undefined;
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
            let limits: GenerationLimits;
            if ("recipes" in derived) {
              recipes = derived.recipes;
              limits = derived.generation;
              // MUST: 入力の検査（値依存 — 毎 run）は backing 構築より前。ここで落ちる run に
              // slot（DiT で ~GiB 規模）の構築を払わせない。
              const data = graph.inputs.map((spec) =>
                this.#checkInput(spec.name, captured.values.get(spec.name), shapes)
              );
              // MUST: backing を作るのは**ヒット run だけ**。単発 run（1 回しか走らない
              // ワークロード）に slot メモリを払わせないための唯一の門で、ミス run の挙動と
              // ArenaStats はこれで完全に据え置かれる。
              // MUST: generation run も**同じ backing に載る**（ADR 0066 決定 5）。載る相手は
              // Session 所有の実体を束ねる dispatch だけで、context 所有の実体を束ねる dispatch は
              // 焼かれずに残り、下の `#generationGroups` が context ごとに埋める。分けているから
              // こそ、backing は context を切り替えても作り直さずに済む。
              const activated = this.#activateBacking(
                preparedKey,
                recipes,
                shapes,
                residentInputs,
              );
              backing = activated.backing;
              builtBacking = activated.built;
              graph.inputs.forEach((spec, index) => {
                // 常駐入力は writeBuffer を出さない（実体がそのまま焼き込まれている）。
                const values = data[index];
                if (values !== undefined) this.#writeInput(activated.backing, spec.name, values);
              });
            } else {
              this.#bindInputs(captured, shapes, arena, env, boundResidents);
              const built = await this.#recipeBuilder.buildRecipes(derived.steps, stateShapes);
              recipes = built.recipes;
              limits = built.generation;
              // MUST: 登録は `RecipeBuilder.buildRecipes` が**完走して戻った後**だけ。途中で throw した run の
              // 部分レシピを載せると、次の同一 bindings の run が欠けたステップ列を実行し、
              // 例外なしで誤った値を返す。
              this.#registerPrepared(preparedKey, {
                shapes,
                recipes,
                fusions: derived.fusions,
                generation: limits,
              });
            }
            this.#lastRunPrepared = {
              hit: prepared !== undefined,
              cachedPlans: this.#state.prepared.size,
            };
            if (generation !== undefined) {
              // MUST: 論理長の検査と搬送は **dispatch を 1 本も積む前**（ここまでの失敗は state に
              // 届かないので poison しない）。`queue.writeBuffer` は issue 順で queue timeline へ
              // 載るので、先行 submit を追い越さない（ADR 0004 不変条件④ / ADR 0066 追記 4）。
              assertGenerationRun(
                limits,
                generation.context.chunkLength,
                pastLength,
                generation.queryLength,
              );
              generation.context[RUNTIME_INTERNAL].writeLengths(
                pastLength,
                generation.queryLength,
              );
            }
            // MUST: スナップショットは書き dispatch を**積む前**（`SubmitScheduler.dispatch` は
            // チャンク上限・時間予算で run の途中に自動 submit する — src/gpu/submit.ts）。
            // 後で取ると、積んだ瞬間の自動 submit を数え損ねて「submit したのに poison
            // しない」= 沈黙破壊になる。逆向きの誤差（積む前の時間予算 submit を数えて
            // しまう過剰 poison）は新しい context で復旧できる安全側。
            // MUST: 2 経路（アリーナ / 焼き込み）で**同じ位置**から呼ぶ。ずらすと同じグラフの
            // 同じ失敗が、ミス run では poison しヒット run ではしない（またはその逆）になる。
            const noteStateWrite = (recipe: StepRecipe): void => {
              if (recipe.writesState && stateWriteSubmits === undefined) {
                stateWriteSubmits = scheduler.submitCount;
              }
            };
            if (backing === undefined) {
              const run = { device, scheduler, arena, env, generation: encoding };
              for (const recipe of recipes) {
                noteStateWrite(recipe);
                executeStepRecipe(recipe, run);
              }
              arena.assertDrained();
            } else {
              // slot 経路。積むコマンド列はアリーナ経路と同一で、bind 先の実体が run を跨いで
              // 固定されるだけ（前 run の残骸が残っていてよい根拠は full-write — ADR 0014）。
              // bind group は構築時に焼き込み済みなので、ここは dispatch を積むだけ
              // （generation run は context 側の束だけを run ごとに照合する — 決定 5）。
              executeBakedPlan(
                recipes,
                backing.groups,
                scheduler,
                generationFace === undefined ? undefined : {
                  groups: this.#generationGroups(generationFace, backing, recipes),
                  encoding: generationFace.encoding,
                  onStep: noteStateWrite,
                },
              );
            }

            if (singleFence) {
              // MUST: 読み戻しの copy は **dispatch と同じコマンド列**へ積む（FIFO）。別 encoder の
              // 別 submit にすると submit が 2 本に割れ、mapAsync が「先行 dispatch の完了」を
              // 含意しなくなる（含意の根拠は同一キューでの発行順）。
              staged = this.#stageOutputs(
                backing?.outputs ?? env,
                shapes,
                arena,
                backing,
                (source, size, staging) => scheduler.copyBuffer(source, 0, staging, 0, size),
              );
              // フェンスを張らずに出し切る。待ちは下の mapAsync 1 本に集約される。
              scheduler.submitPending();
            } else {
              await scheduler.flush();
            }
            // errorScope の網は経路で変えない（単一フェンス経路では readback の copy と staging
            // 確保も run 本体の 2 本に相乗りする — 二段待ち経路の `#readOutputs` が張る対と同じ
            // `out-of-memory` + `validation`）。
            const pending = popFailureScopes(
              device,
              singleFence ? "run のエンコードと readback" : "run のエンコード",
            );
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
          outputs = staged === undefined
            ? await this.#readOutputs(backing?.outputs ?? env, shapes, arena, backing)
            : await this.#finishStagedRead(staged);
          this.#lastRun = arena.stats;
          this.#lastRunParams = this.#recipeBuilder.paramsStats;
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
        } finally {
          // MUST: 束縛予約は成功・失敗のどちらの経路でも必ずここで返す（返し損ねるとその常駐
          // テンソルは Session の寿命いっぱい破棄できなくなる）。この時点では読み戻しまで含めて
          // 全ての GPU 操作が submit 済みなので、以後の `dispose()` は「submit 済みコマンドが
          // 参照するバッファの破棄」= WebGPU 的に安全（実解放は完了まで実装が遅延する）。
          releaseBoundResidents(boundResidents);
        }
        // 本体が通ったときは後始末の失敗をそのまま伝える（flush 未完了のまま返さない）。
        await arena.destroy();
        // 切替 / 追い出しで浮いた旧 backing を返す唯一の後始末点（flush 後 — ADR 0004 の
        // flush-before-destroy）。dispose が同じことをするので、ここを通らずに落ちた run の
        // 積み残しも取りこぼさない。
        this.#destroyRetired();
        return outputs;
      });

    try {
      const outputs = await encode();
      // MUST: 論理長を進めるのは run が**例外なく返った**ときだけ（ADR 0066 決定 6 —
      // 「論理長は run の成功で進む」の成功はこの意味）。readback や後始末で落ちた run は
      // 物理 ring だけが進んだ状態なので、進めずに下の poison へ倒す。
      if (generation !== undefined) {
        generation.context[RUNTIME_INTERNAL].advance(pastLength, generation.queryLength);
      }
      return outputs;
    } catch (cause) {
      this.#poisonOnStateWrite(generation, stateWriteSubmits, cause);
      throw cause;
    }
  }

  /**
   * 失敗した generation run の後始末（ADR 0066 追記 3 の「失敗の原子性」）。
   *
   * state 変更 dispatch を**submit した**なら物理 ring は上書きされ得るが論理長は進んでいない
   * ので、context は poison する（rollback / staging は持たない設計 — 復旧は新しい context +
   * ホスト側の state 再構築）。submit していなければ pending は `scheduler.discard()` が捨てて
   * おり ring は無傷なので、poison しない。
   *
   * MUST: 判定不能な経路は poison に倒す（過剰 poison は新しい context で復旧できるが、
   * 過少 poison は「壊れた KV で生成が続く」沈黙破壊になる）。スナップショットを書き
   * dispatch の**前**に取っているのはそのため。
   */
  #poisonOnStateWrite(
    generation: GenerationRun | undefined,
    snapshot: number | undefined,
    cause: unknown,
  ): void {
    if (generation === undefined || snapshot === undefined) return;
    if (this.#state.scheduler.submitCount === snapshot) return;
    generation.context[RUNTIME_INTERNAL].poison(
      `state 変更 dispatch を submit した run が失敗した（${
        cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause)
      }）`,
    );
  }

  /**
   * アリーナ経路の入力束縛をグラフ入力ぶんまとめて積む（{@link Session.#bindInput} の呼び口）。
   * 非 backed のミス run と generation run が共有する 1 本。
   */
  #bindInputs(
    captured: CapturedInputs,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    env: Map<string, GPUBuffer>,
    bound: ResidentTensor[],
  ): void {
    for (const spec of this.#state.graph.inputs) {
      const value = captured.values.get(spec.name);
      env.set(spec.name, this.#bindInput(spec.name, value, shapes, arena, bound));
    }
  }

  /**
   * enqueue 1 本（{@link Session.enqueue} の本体）。
   *
   * 構造は backed run の中身そのままから「フェンスと readback を伴う部分」を全部落とした形:
   *
   * - **errorScope を張らない**。batch が `out-of-memory` + `validation` の 2 本を区間ぶん
   *   張り続けており、ここで push すると LIFO の入れ子が enqueue の本数だけ深くなる（しかも
   *   pop は await を跨ぐので区間ロックが要る = batch が保持中で自己デッドロック）。
   * - **`GpuContext` の `withScopeLock` を取らない**。区間ロックは batch が握っている
   *   （取りに行くと自己デッドロック — GpuContext の不変条件）。
   * - **{@link RunArena} を作らない**。通る経路は slot backing だけで、run 寿命の確保が
   *   1 バイトも出ない（readback staging すら出ない）。
   *
   * 入力は run と同じく**発行時の写し**（{@link CapturedInputs}）で受け取る — ここも
   * 1 マイクロタスク以降に走るため、利用者の Record を引き直すと発行後の書き換えを読む。
   *
   * MUST: 初回（導出済み計画が無い）でも backing を作る。run は「ヒット run だけ backing を
   * 作る」— 単発 run に slot メモリを払わせないため — が、enqueue は最初から繰り返し前提の面
   * なので、この門を持ち込むと 1 本目が非 backed（= フェンスを伴うアリーナ経路）に落ちる。
   * 黙って落とさないのがこの面の契約なので、初回はここで払う。
   */
  async #enqueueOnce(captured: CapturedInputs, options: EnqueueOptions): Promise<void> {
    const { graph, scheduler } = this.#state;
    // 受け口の検査とリース取得は {@link Session.enqueue} の同期区間で済んでいる（本体で
    // やると finish との競走が閉じない）。ここは決着の相手として登録するだけ。
    // MUST: 区間の決着で「未 submit を出し切る」「計測窓を閉じる」の相手として登録する。
    options.batch[RUNTIME_INTERNAL].join(scheduler);

    const { inputShapes, residentInputs } = captured;
    // MUST: 束縛の解決（= 入力 shape の検証）は run と同じく毎回走らせる。
    const resolved = bindSymbols(
      graph,
      inputShapes,
      options.bindings ?? {},
      residentNames(residentInputs),
    );
    // MUST: `enqueue` は generation 面を持たない（波 D-5）。state 参照グラフは
    // `stateShapes` 無しの導出で fail loudly になる（黙って state 抜きで走らない）。
    const preparedKey = this.#preparedKey(resolved, residentInputs, undefined);
    const prepared = this.#takePrepared(preparedKey);
    const derived = prepared ?? this.#planSteps(resolved, undefined);
    const shapes = derived.shapes;
    this.#lastRunFusions = derived.fusions;
    this.#lastRunPrepared = undefined;
    this.#recipeBuilder.resetParamsStats();

    let builtBacking = false;
    try {
      let recipes: readonly StepRecipe[];
      if ("recipes" in derived) {
        recipes = derived.recipes;
      } else {
        const built = await this.#recipeBuilder.buildRecipes(derived.steps);
        recipes = built.recipes;
        // MUST: 登録は `RecipeBuilder.buildRecipes` が完走して戻った後だけ（run と同じ理由）。
        this.#registerPrepared(preparedKey, {
          shapes,
          recipes,
          fusions: derived.fusions,
          generation: built.generation,
        });
      }
      // MUST: 入力と写し先の検査（値依存）は backing 構築より前。ここで落ちる enqueue に
      // slot の構築を払わせない。
      const data = graph.inputs.map((spec) =>
        this.#checkInput(spec.name, captured.values.get(spec.name), shapes)
      );
      const copies = this.#planCopyOutputs(options.copyOutputs, shapes);
      const activated = this.#activateBacking(preparedKey, recipes, shapes, residentInputs);
      builtBacking = activated.built;
      // MUST: 写し元の解決（実体に依存する検査）は dispatch を 1 本も積む前に済ませる。
      const writes = this.#resolveCopyOutputs(copies, activated.backing);
      graph.inputs.forEach((spec, index) => {
        const values = data[index];
        if (values !== undefined) this.#writeInput(activated.backing, spec.name, values);
      });
      this.#lastRunPrepared = {
        hit: prepared !== undefined,
        cachedPlans: this.#state.prepared.size,
      };
      executeBakedPlan(recipes, activated.backing.groups, scheduler);
      // MUST: 写しは dispatch 列の**後**に積む（同じコマンド列の FIFO が「書き終わった slot を
      // 読む」の根拠）。写し先が同じ enqueue の常駐入力を兼ねる形（ループの状態更新）も、
      // 読む dispatch が全て先に積まれているので正しい。
      for (const write of writes) {
        scheduler.copyBuffer(write.source, 0, write.target, 0, write.size);
      }
      // MUST: 末尾で必ず submit する。以後の `queue.writeBuffer`（次の enqueue の入力書き込み）は
      // queue timeline へ issue 順に載るので先行 dispatch を追い越さない — 追い越すのは
      // **未 submit の**エンコードだけ（ADR 0004 不変条件④）。pending を残す経路を作ると、
      // 次の入力が前の enqueue に読まれる沈黙誤値になる。
      scheduler.submitPending();
    } catch (cause) {
      // MUST: 失敗した enqueue の残 pending は submit せずに捨てる（run と同じ規律）。
      scheduler.discard();
      if (builtBacking) this.#retireBacking();
      this.#destroyRetired();
      throw cause;
    }
    // MUST: 破棄待ちを返してよいのは submit / discard の**後**だけ（未 submit のエンコードが
    // 破棄済みバッファを参照しないこと）。submit 済みコマンドからの参照は WebGPU 的に安全で、
    // 実解放は完了まで実装が遅延する。
    this.#destroyRetired();
    // アリーナを 1 度も作らないので、直近 run の中間バッファ実績は「無い」が正しい
    // （前の run のものを残すと、enqueue の後に読んだ診断が別の実行の話になる）。
    this.#lastRun = undefined;
    this.#lastRunParams = this.#recipeBuilder.paramsStats;
  }

  /**
   * `copyOutputs` の検査（**backing 構築より前**に済ませる）。グラフ出力であること・常駐先が
   * 生きていること・大きさが宣言 shape ぶんと厳密一致であることを見る。
   */
  #planCopyOutputs(
    copyOutputs: Readonly<Record<string, ResidentTensor>> | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): readonly PlannedCopy[] {
    if (copyOutputs === undefined) return [];
    return Object.entries(copyOutputs).map(([name, target]) => {
      if (!this.#state.outputNames.has(name)) {
        throw new ExecutionError(
          `copyOutputs '${name}' はグラフ出力ではない（[${
            [...this.#state.outputNames].join(", ")
          }]）`,
        );
      }
      assertResidentUsable(target, this.#state.gpu, `copyOutputs '${name}'`);
      const shape = resolvedShape(shapes, name);
      // MUST: 出力 slot の大きさと同じ算式（`#readOutputs` の staging と揃える）。
      const size = Math.max(4, numel(shape) * 4);
      if (target.byteLength !== size) {
        throw new ExecutionError(
          `copyOutputs '${name}': 常駐テンソル '${target.label}' の ${target.byteLength} バイトが shape [${
            shape.join(",")
          }] の ${size} バイトと合わない`,
        );
      }
      return { name, target, size };
    });
  }

  /**
   * `copyOutputs` の写し元（グラフ出力の実体）を解決する。実体に依存する検査なので backing の
   * 活性化より後だが、**dispatch を 1 本も積む前**に済ませる。
   *
   * MUST: 写し元と写し先が同一バッファになる形はここで落とす。グラフ出力が入力の別名
   * （`reshape` / 恒等 `expand`）になる縮退グラフでは、出力の実体が常駐入力そのものになり、
   * 自己コピーが積まれる。これは WebGPU の validation に捕まるが、**捕まるのは
   * `batch.finish()`** なので ①原因の enqueue が特定できず ②同じ区間の無関係な enqueue まで
   * 巻き添えで落ちる（失敗の帰属は batch 単位 — BatchScope の設計上の受容）。真因を名指しで
   * enqueue の呼び出し点へ返すために、ここで 1 行検査する。
   */
  #resolveCopyOutputs(
    copies: readonly PlannedCopy[],
    backing: ActiveBacking,
  ): readonly ResolvedCopy[] {
    return copies.map((copy) => {
      const source = backing.outputs.get(copy.name);
      // IR は graph.outputs に initializer 名を書くことを許しており、その値は焼き込みの
      // 値写像に載らない（run 側は `#readOutputs` の重みフォールバックが受け持つ）。写しの
      // 相手にする意味は無い（実行のたびに同じ定数を GPU 内でコピーするだけ）ので落とす。
      if (source === undefined) {
        throw new ExecutionError(
          `copyOutputs '${copy.name}': ノード出力ではない（initializer をそのままグラフ出力に` +
            "した値は写せない — その値は実行に依らず不変)",
        );
      }
      const target = copy.target[RUNTIME_INTERNAL].buffer;
      if (source === target) {
        throw new ExecutionError(
          `copyOutputs '${copy.name}': 写し元と写し先が同じバッファ（常駐テンソル ` +
            `'${copy.target.label}'）— グラフ出力がその常駐入力の別名（reshape / 恒等 expand）に` +
            "なっているため、写しは自己コピーになる（計算内容がゼロの構成）",
        );
      }
      return { source, target, size: copy.size };
    });
  }

  /**
   * 導出相の前半（計画 → 融合判定）。GPU に触れない純関数だけで閉じる。
   *
   * device の limits は**値として**渡す（融合ルールが GPUDevice を掴まないまま、行ブロック
   * 枚数のような「機の能力で決まる計画」を静的に決められる — src/runtime/fusion.ts の
   * {@link FusionLimits}）。
   */
  #planSteps(
    bindings: SymbolBindings,
    stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  ): PlannedSteps {
    const plan = planGraph(this.#state.graph, bindings, stateShapes);
    const { maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension } =
      this.#state.gpu.limits;
    const fusion = planFusions(plan.nodes, {
      useCounts: this.#state.useCounts,
      outputNames: this.#state.outputNames,
      limits: { maxStorageBufferBindingSize, maxComputeWorkgroupsPerDimension },
      ...(this.#state.rowBlockSplit === undefined
        ? {}
        : { rowBlockSplit: this.#state.rowBlockSplit }),
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
   * MUST: **常駐入力の識別子も載せる**。導出結果（レシピ）自体は bindings だけの関数だが、
   * この鍵は slot backing の signature でもあり、backing は焼き込み時に常駐入力の実体を
   * bind group へ畳み込む — 差し替えを鍵に反映しないと、**前の常駐テンソルを束ねたままの
   * bind group** で回り続ける（例外は 1 つも出ず、値だけが古い）。区切りの `|` は bindings
   * 側の連結（数値とカンマだけ）には現れないので、常駐入力の無い run の鍵は従来と同一のまま。
   * MUST: **{@link GenerationContext} の識別子は載せない**（ADR 0066 決定 5）。載せると context を
   * 切り替えるたびに計画・融合判定・レシピ導出と slot backing 構築が全滅する（decode の
   * ホットパスで毎シーケンス再導出になる）。常駐入力と逆の扱いになるのは、context 所有の実体
   * （state スロットと論理長 uniform）を Session 所有の bind group へ焼き込まないため — 分離の
   * 代償として「state を含む bind group」だけを context 側で束ね直す
   * （{@link Session.#generationGroups}）。
   * この不変条件は tests/gpu_generation_context_test.ts が「context を 2 本作っても導出済み計画と
   * params キャッシュが増えない」形で門にしている。
   * MUST: 載せるのは**解決済みスロット容量**の側（ADR 0066 決定 3 の「鍵は容量」・追記 7）。
   * 同じ Session で C=512 と C=131072 の context を作れば、state 参照計画は容量ごとに別鍵に
   * なるのが正しい（レシピは容量を params と S の確保サイズへ焼き込むため）。区切りの `#` は
   * 上流 2 節（数値・カンマ・`|`・`t`/`r<id>`）に現れないので、**generation を伴わない run の
   * 鍵は 1 文字も動かない**。
   */
  #preparedKey(
    bindings: SymbolBindings,
    residents: ReadonlyMap<string, ResidentTensor>,
    stateShapes: ReadonlyMap<string, readonly number[]> | undefined,
  ): string {
    const graph = this.#state.graph;
    // states 専用記号（KV 容量 `C` 等）は dims 節に**載せない** — bindSymbols が要求も受理も
    // しない（ADR 0066 追記 7）ので値が無く、容量の情報は下の capacities 節が完全に持つ。
    const statesOnly = statesOnlySymbols(graph);
    const dims = graph.symbols.filter((sym) => !statesOnly.has(sym))
      .map((sym) => bindings[sym]).join(",");
    const bound = residents.size === 0 ? "" : `|${
      graph.inputs
        .map((spec) => {
          const resident = residents.get(spec.name);
          return resident === undefined ? "t" : `r${resident[RUNTIME_INTERNAL].id}`;
        })
        .join(",")
    }`;
    if (stateShapes === undefined) return `${dims}${bound}`;
    // MUST: 並べる順は `graph.states` の宣言順（Map の反復順に依存させない — スロットの
    // 集合が同じでも順序が違う 2 つの context が別鍵になると、切替のたびに再導出が起きる）。
    const capacities = Object.keys(graph.states)
      .map((name) => (stateShapes.get(name) ?? []).join("x"))
      .join(";");
    return `${dims}${bound}#${capacities}`;
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
   * MUST: 確保から `this.#backing` への代入（= 所有権の確立）までを try/catch で囲み、途中の
   * 同期例外では確保済みを `#retired` へ回す。この窓で漏れた実体は `#retireBacking()` からも
   * `Session.dispose()` からも到達できず、しかも量はこの Session で最大（slot 表の総バイト）に
   * なる（ADR 0004「確保と破棄を 1 箇所へ」は失敗経路でも保つ）。
   * NOTE: run 経路のレシピ列は必ず 1 度アリーナ経路で完走している（run は**ヒット run でしか**
   * backing を作らない）ので、参照計数が閉じることは `assertDrained` が確かめ済み。
   * `#enqueueOnce` は MUST として**初回から** backing を作るのでその前提は成り立たないが、
   * 破れは `derivePlanSlots` 自身の参照計数検査（負値で即 throw）が受け持つ。
   */
  #activateBacking(
    key: string,
    recipes: readonly StepRecipe[],
    shapes: ReadonlyMap<string, readonly number[]>,
    residentInputs: ReadonlyMap<string, ResidentTensor>,
  ): { readonly backing: ActiveBacking; readonly built: boolean } {
    const current = this.#backing;
    if (current !== undefined && current.key === key) return { backing: current, built: false };
    // MUST: 旧 backing は destroy せず破棄待ちへ積む（この run の flush 後にだけ返す）。
    this.#retireBacking();
    const graph = this.#state.graph;
    const device = this.#state.gpu.device;
    const slots = derivePlanSlots(recipes);
    // 所有権が確立する（`this.#backing` への代入）までの確保物と retain 済み常駐入力。
    // 途中で同期例外が出たら catch がここから回収する。
    const buffers: GPUBuffer[] = [];
    const ownedInputs: GPUBuffer[] = [];
    const retained: ResidentTensor[] = [];
    try {
      for (const size of slots.bytes) {
        buffers.push(device.createBuffer({ size, usage: STORAGE_USAGE }));
      }
      // 通常入力のバッファは backing 所有にする（run ごとの確保と writeBuffer 先の入れ替わりを
      // 消す）。常駐入力は GpuContext 所有の実体をそのまま束ね、**所有しない**。
      // MUST: 大きさはアリーナ経路の `#bindInput` と同じ算式（4 バイト床込み — 0 要素入力で
      // 0 サイズバッファを束縛しない）。同一 signature なら不変。
      const inputs = new Map<string, GPUBuffer>(
        graph.inputs.map((spec) => {
          const resident = residentInputs.get(spec.name);
          if (resident !== undefined) return [spec.name, resident[RUNTIME_INTERNAL].buffer];
          const buffer = device.createBuffer({
            size: Math.max(4, numel(resolvedShape(shapes, spec.name)) * 4),
            usage: HOST_WRITTEN_USAGE,
          });
          ownedInputs.push(buffer);
          return [spec.name, buffer];
        }),
      );
      const baked = bakeBindGroups(recipes, slots, { device, buffers, inputs });
      // 読み戻し先はここで確定する。initializer がグラフ出力になる形（IR が許す）は値名の
      // 写像に載らないので、`#readOutputs` 側の重みフォールバックがそのまま受け持つ。
      const outputs = new Map<string, GPUBuffer>();
      for (const name of graph.outputs) {
        const buffer = baked.values.get(name);
        if (buffer !== undefined) outputs.set(name, buffer);
      }
      const residents = [...residentInputs.values()];
      // MUST: 焼き込みの参照は backing が生きている間ずっと積んでおく（退役で返す）。これが
      // 「参照中の ResidentTensor.dispose を fail loudly にする」唯一の根拠。
      for (const resident of residents) {
        resident[RUNTIME_INTERNAL].retainBaked();
        retained.push(resident);
      }
      // 世代識別子は「作った順」の単調カウンタそのもの（診断の `buildCount` と同じ採番）。
      // MUST: 識別子と累計を**同じ 1 つの式**から採る（別々に足すと、片方だけ通らない経路で
      // 恒久的にずれて「作り直したのに同じ世代」になる）。
      const build = this.#backingBuilds + 1;
      const backing: ActiveBacking = {
        key,
        build,
        bytes: slots.bytes.reduce((total, size) => total + size, 0),
        inputs,
        residents,
        groups: baked.groups,
        slots,
        buffers,
        outputs,
        owned: new Set([...buffers, ...ownedInputs]),
        readable: new Set([
          ...[...slots.pinned].map((slot) => buffers[slot]),
          ...inputs.values(),
        ]),
      };
      this.#backing = backing;
      this.#backingBuilds = build;
      return { backing, built: true };
    } catch (cause) {
      // MUST: 確保物は既存の破棄経路（`#retired` → `#destroyRetired`）へ載せるだけにする。
      // ここで destroy すると破棄の担い手が 2 つになる（ADR 0004）。この時点では bind group を
      // 使う dispatch が 1 本も積まれていない（`#activateBacking` は dispatch より前）ので、
      // 後始末点での破棄が submit 済みコマンドを壊すことはない。
      for (const buffer of buffers) this.#retired.push(buffer);
      for (const buffer of ownedInputs) this.#retired.push(buffer);
      for (const resident of retained) resident[RUNTIME_INTERNAL].releaseBaked();
      throw cause;
    }
  }

  /**
   * generation run の **context 側** bind group を用意する（ADR 0066 決定 5 の分離焼き込み）。
   *
   * 束の寿命は **(context, backing 実体) の組**なので、判別は backing の世代識別子 1 つで足りる
   * （context 側は自分のキャッシュしか見ない）。同一 context の連続 run は 0 回・context 切替と
   * backing 再構築でだけ 1 回焼く形になり、その回数が診断 `stateBacking.rebindCount` になる。
   *
   * MUST: 「世代識別子の照合 → 焼き直し → dispatch」の順を**この 1 本に閉じる**。照合を経ずに
   * キャッシュを読める口を別に作ると、退役した backing の実体を束ねた古い group で dispatch する
   * 形が生まれる。退役した実体は**その run の後始末まで生きている**（`#retired` → flush 後の
   * `#destroyRetired`）ので、古い束は別の計画の実体を束ねたまま validation を通り、**値だけが
   * 静かに変わる**（照合を外した故障注入で実測 — 出力 64 要素が全て不一致・例外ゼロ）。
   * MUST: 呼ぶのは run の errorScope 区間の内側だけ（createBindGroup の validation 失敗は
   * 例外にならない）。
   * NOTE: 束に載るのは `encoding` の**バッファだけ**（`past` / `query` は uniform の中身）。
   * だから毎 run 論理長が変わっても束は無効にならず、焼き直しの条件は backing の世代だけになる。
   */
  #generationGroups(
    face: { readonly context: GenerationContext; readonly encoding: GenerationEncoding },
    backing: ActiveBacking,
    recipes: readonly StepRecipe[],
  ): BakedGroups {
    const internals = face.context[RUNTIME_INTERNAL];
    const cached = internals.bakedGroups(backing.build);
    if (cached !== undefined) return cached;
    const groups = bakeGenerationBindGroups(recipes, backing.slots, {
      device: this.#state.gpu.device,
      // MUST: backing が焼き込みに使ったものをそのまま渡す（同じ値解決の再現 —
      // {@link bakeGenerationBindGroups}）。
      buffers: backing.buffers,
      inputs: backing.inputs,
      generation: face.encoding,
    });
    internals.setBakedGroups(backing.build, groups);
    this.#stateRebinds += 1;
    return groups;
  }

  /**
   * 活性 backing を破棄待ちへ移す（実際の `destroy()` は flush / submit 後の後始末点で 1 回だけ）。
   *
   * 常駐入力の焼き込み参照はここで返す。返してよいのは、退役した backing の bind group を
   * 使う dispatch がこれ以降 1 本も積まれないため（積むのは活性 backing だけ）。
   */
  #retireBacking(): void {
    const current = this.#backing;
    if (current === undefined) return;
    for (const buffer of current.owned) this.#retired.push(buffer);
    for (const resident of current.residents) resident[RUNTIME_INTERNAL].releaseBaked();
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

  /** 値の宣言 dtype。引けないのはランタイム内部の不変条件破れ（declaredDtypes は全値を持つ）。 */
  #declaredDtype(name: string): IrDtype {
    const dtype = this.#state.dtypes.get(name);
    if (dtype === undefined) {
      throw new ExecutionError(`値 '${name}' の宣言 dtype が無い`);
    }
    return dtype;
  }

  /**
   * 入力の検査（**値依存なので毎 run 必要** — ヒット run でも飛ばせない）。通れば GPU へ書く
   * ホスト配列を返し、**常駐入力では `undefined`**（書くものが無い）を返す。
   */
  #checkInput(
    name: string,
    value: RunInput | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): Tensor["data"] | undefined {
    if (value instanceof ResidentTensor) {
      this.#checkResidentInput(name, value, shapes);
      return undefined;
    }
    return this.#checkTensorInput(name, value, shapes);
  }

  /** ホスト配列の入力検査。通れば GPU へ書く配列をそのまま返す。 */
  #checkTensorInput(
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

  /**
   * 常駐入力の検査。
   *
   * MUST: 常駐テンソルは dtype を持たない（バイト列と大きさだけの寿命クラス）ので、検査
   * できるのは大きさだけ。**厳密一致**を要求する — 大きい常駐テンソルの先頭だけを束ねる形を
   * 許すと、`arrayLength()` が束縛範囲のバイト数から決まる WGSL 側で要素数が静かに変わる
   * （アリーナが「要求より大きいバッファを配らない」のと同じ理由）。
   */
  #checkResidentInput(
    name: string,
    resident: ResidentTensor,
    shapes: ReadonlyMap<string, readonly number[]>,
  ): void {
    assertResidentUsable(resident, this.#state.gpu, `入力 '${name}'`);
    const shape = resolvedShape(shapes, name);
    const wanted = Math.max(4, numel(shape) * 4);
    if (resident.byteLength !== wanted) {
      throw new ExecutionError(
        `入力 '${name}': 常駐テンソル '${resident.label}' の ${resident.byteLength} バイトが shape [${
          shape.join(",")
        }] の ${wanted} バイトと合わない`,
      );
    }
  }

  /**
   * アリーナ経路の入力束縛。通常入力は run 寿命のバッファを 1 本確保して書き、常駐入力は
   * GpuContext 所有の実体をそのまま返す（**確保も writeBuffer も出ない**）。
   *
   * MUST: 常駐入力は束縛と同時に予約を 1 本積み、`bound` へ載せる。この経路（ミス run）は
   * 「env へ生バッファを束縛 → `buildRecipes` を await → エンコード」の順で進み、焼き込み参照は
   * まだ 1 本も無い。await の窓で `dispose()` が来ると素通りしてしまい、再開したエンコードが
   * 破棄済みバッファを掴む（run は errorScope に捕まって落ちるが、真因から遠い）。予約が
   * あれば誤りは dispose の呼び出し点で {@link ResidentTensorError} として即座に落ちる。
   */
  #bindInput(
    name: string,
    value: RunInput | undefined,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    bound: ResidentTensor[],
  ): GPUBuffer {
    if (value instanceof ResidentTensor) {
      this.#checkResidentInput(name, value, shapes);
      value[RUNTIME_INTERNAL].retainBound();
      bound.push(value);
      return value[RUNTIME_INTERNAL].buffer;
    }
    const data = this.#checkTensorInput(name, value, shapes);
    const buffer = arena.allocHostWritten(Math.max(4, data.length * 4), HOST_WRITTEN_USAGE);
    if (data.length > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /**
   * backed run の入力書き込み（backing 所有の常駐バッファへ**上書き**する）。
   *
   * MUST: この `writeBuffer` が追い越せる未 submit エンコードは存在しない。根拠は 3 つ —
   * ①同一 Session の run / enqueue / dispose は {@link Session.#chain} で直列化される
   * ②先行実行は戻る時点で未 submit のエンコードを 1 つも残していない: run は readback の完了
   * （二段待ち経路は flush の `onSubmittedWorkDone` も）まで済ませてから返り、**enqueue は
   * 末尾で必ず eager submit する**（{@link SubmitScheduler.submitPending}）③実行の中では入力
   * 書き込みが全エンコードに先行する。`queue.writeBuffer` は queue timeline へ issue 順に載るので
   * **submit 済み**の先行 dispatch は追い越さず、追い越すのは未 submit のエンコードだけ
   * （ADR 0004 不変条件④）。どれかが崩れると前の実行の dispatch が新しい入力を読む沈黙誤値に
   * なる。したがって「enqueue 後に submit せず pending を残す経路」を作ってはならない。
   */
  #writeInput(backing: ActiveBacking, name: string, data: Tensor["data"]): void {
    const buffer = backing.inputs.get(name);
    if (buffer === undefined) throw new ExecutionError(`入力 '${name}' の常駐バッファが無い`);
    if (data.length > 0) this.#state.gpu.device.queue.writeBuffer(buffer, 0, data);
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

  /**
   * グラフ出力ごとに staging を確保し、そこへの copy を `copy` に積ませる（**同期**）。
   *
   * MUST: 読み戻すのはグラフ出力のみ。中間値はプール再利用で内容が入れ替わっている。
   * NOTE: 積み先を引数に取るのは、単一フェンス経路（run 本体のコマンド列 —
   * {@link SubmitScheduler.copyBuffer}）と二段待ち経路（readback 専用 encoder）で**積む先だけ**
   * が違うため。staging の作り方と読み戻し適格の判定を経路ごとに 2 本持たない。
   */
  #stageOutputs(
    env: ReadonlyMap<string, GPUBuffer>,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    backing: ActiveBacking | undefined,
    copy: (source: GPUBuffer, size: number, staging: GPUBuffer) => void,
  ): readonly StagedOutput[] {
    const staged: StagedOutput[] = [];
    for (const name of this.#state.graph.outputs) {
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
      copy(buffer, size, staging);
    }
    return staged;
  }

  /**
   * staging の map 完了を待つ（**このバッファへの最後の使用 = 積んだ copy の完了**が解決条件）。
   *
   * MUST: 複数出力は**並列**に待つ。直列 await にすると待ちの固定費を出力数ぶん払う。
   * MUST: 消失後の mapAsync が解決しない実装がありうる（実測は raceCanaryDeviceLost の doc）
   * ため競わせる — ハングを失敗に変換する保険。
   */
  async #awaitStaged(staged: readonly StagedOutput[]): Promise<void> {
    await this.#state.gpu[RUNTIME_INTERNAL].raceDeviceLost(
      Promise.all(staged.map((item) => item.staging.mapAsync(MAP_MODE.READ))),
      "readback",
    );
  }

  /** map 済み staging をホストの {@link Tensor} へ組み直す（同期・unmap まで含む）。 */
  #collectStaged(staged: readonly StagedOutput[]): RunOutputs {
    // MUST: グラフ出力名由来のキーを蓄積する器は null プロトタイプ。素の `{}` では
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
  }

  /**
   * 単一フェンス経路の読み戻し（H-1）。run 本体のコマンド列に積んだ copy が唯一の待ち相手で、
   * **この mapAsync が run のフェンスそのもの**になる。
   */
  async #finishStagedRead(staged: readonly StagedOutput[]): Promise<RunOutputs> {
    await this.#awaitStaged(staged);
    // MUST: 計測窓を閉じるのはフェンスの**後**（前に閉じると実測が過小に出てチャンクが膨らみ、
    // TDR 域へ向かう危険側 — src/gpu/submit.ts）。窓にはホストの後始末まで含むので推定は過大
    // 側に出るが、向きは安全側で据え置き。
    this.#state.scheduler.closeMeasurementWindowAfterFence();
    return this.#collectStaged(staged);
  }

  /**
   * 二段待ち経路の読み戻し（gpuTiming 有効時 / グラフ出力 0 本 — {@link Session.#runOnce} の
   * `singleFence`）。run 本体の flush とは**別 encoder の別 submit**で、フェンスは
   * `onSubmittedWorkDone` に続く 2 本目になる。
   */
  async #readOutputs(
    env: ReadonlyMap<string, GPUBuffer>,
    shapes: ReadonlyMap<string, readonly number[]>,
    arena: RunArena,
    backing: ActiveBacking | undefined,
  ): Promise<RunOutputs> {
    const device = this.#state.gpu.device;
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
      const staged = this.#stageOutputs(
        env,
        shapes,
        arena,
        backing,
        (source, size, staging) => encoder.copyBufferToBuffer(source, 0, staging, 0, size),
      );
      device.queue.submit([encoder.finish()]);

      // MUST: mapAsync より前に pop を発行する（mapAsync 待ちの間に別の操作を吸わないため）。
      // mapAsync 自身はどのスコープにも入らないが、#runOnce のロック区間内なので、その失敗が
      // 他人のスコープに帰属することはない（run の GPU 操作は全てロック区間内 — 上記不変条件）。
      const pending = popFailureScopes(device, "出力の readback");
      popped = true;
      const failure = await pending;
      if (failure !== undefined) throw failure;

      await this.#awaitStaged(staged);
      return this.#collectStaged(staged);
    } finally {
      // `!popped` は本体が例外で抜けたときにだけ成立する。後始末でその例外を隠さない。
      // staging の破棄は RunArena が持つ（成功・失敗のどちらの経路でも #runOnce が
      // arena.destroy() まで必ず進む — ADR 0004「確保と破棄を 1 箇所へ」）。
      if (!popped) await discardFailureScopes(device);
    }
  }
}

/**
 * 重み shard が 1 本も無い列（全量面 = 全テンソルがグラフ shard に同居した列）。
 *
 * MUST: 呼ぶたびに新しい iterator を返す（使い切った generator を使い回すと、2 本目の
 * Session 構築が「既に done」の列を受けたのか空列なのか区別できない）。
 */
const noWeightShards = (): AsyncIterable<ModelShard> => ({
  [Symbol.asyncIterator]: () => ({
    next: (): Promise<IteratorResult<ModelShard, undefined>> =>
      Promise.resolve({ done: true, value: undefined }),
  }),
});

/**
 * 重み DL 前の admission 相の成果物（ADR 0070 決定 5 / graph-first）— グラフ shard だけで
 * 「実行できるか」を決め、必要メモリを見積り、そのまま Session 構築の入口になる。
 *
 * 保持するのは ①parse 済みグラフ shard（小テンソルの正本）②`IrGraph` ③常駐計画（席）の 3 つ
 * だけで、**GPU 資源は一切持たない**（`estimate` は純関数のまま — 決定 5 の「GPU 非依存」）。
 * グラフ shard のバイト列を createSession まで抱えるのは ADR 0070 決定 3 がグラフ shard を
 * 「karume_ir + 小テンソル」と規定しているからで、RAM ピーク目標「O(最大**重み** shard)」は
 * 崩れない（全量面は元から呼び手が `KarumeModel` を持っているので増分ゼロ）。
 *
 * MUST: 構築は {@link PreparedModel.createSession} だけを入口にするため、mod.ts では**型として
 * のみ**公開する（`Session` / `GpuContext` と同じ流儀 — 直接構築すると capability 門と
 * 契約検査を迂回した「実行できないモデルの Session」が作れてしまう）。
 */
export class PreparedModel {
  /** グラフ shard の parse 結果（小テンソルの正本 — validator.intake が重みとして消費する）。 */
  readonly #file: SafetensorsFile;
  readonly #graph: IrGraph;
  /**
   * 失敗とフェンスの帰属先。**全量面は undefined**（帰属先が 1 つしかない単一ファイル面の
   * 文言を変えない MUST — ADR 0070 受入①の契約）。
   */
  readonly #origin: string | undefined;
  /** 席とバイト数（グラフだけで決まる）— 見積りと構築が共有する 1 個。 */
  readonly #residency: ReadonlyMap<string, WeightResidency>;

  private constructor(file: SafetensorsFile, graph: IrGraph, origin: string | undefined) {
    // 非対応 op / 格納 dtype は全件列挙して落とす（1 件ずつ落とすと何本足りないか分からない）。
    // MUST: この 2 門は**重み shard に触れる前**に通す（2 段境界の存在理由そのもの — 実行
    // できないモデルの重みを DL してから落とすことがなくなる）。
    // NOTE: capability 不足も契約違反もモデル（グラフ）の性質で shard 由来ではないので帰属を
    // 足さない — 全量面と同じ文言のままにする。
    assertRuntimeSupport(graph, RUNTIME_SUPPORT);
    validateGraphContracts(graph);
    this.#file = file;
    this.#graph = graph;
    this.#origin = origin;
    this.#residency = planWeightResidency(graph);
  }

  /** グラフ shard（`__metadata__.karume_ir` 持ちの先頭 shard）から prepare する。 */
  static fromGraphShard(graphShard: ModelShard): PreparedModel {
    const origin = shardOrigin(0, graphShard.id);
    const file = parseShard(graphShard.bytes, origin);
    let graph: IrGraph;
    try {
      graph = extractIrGraph(file);
    } catch (cause) {
      // 「先頭がグラフ shard でない」も shard の取り違え — どの資産を先頭に置いたのかを名乗る。
      throw attributeToShard(origin, cause);
    }
    return new PreparedModel(file, graph, origin);
  }

  /** 全量面（openModel 済み）から prepare する（= 全テンソル同居のグラフ shard 1 本）。 */
  static fromModel(model: KarumeModel): PreparedModel {
    return new PreparedModel(model.file, model.graph, undefined);
  }

  /**
   * グラフの**宣言**（入出力・記号次元・値の shape）。
   *
   * 呼び手（models のパイプライン）は「グラフ宣言と自分の設定の突合」を admission 相で
   * 済ませたい — グラフ shard しか手元に無い段階では `openModel`（全量 1 本を前提に宣言の
   * 完全性まで見る面）が使えないので、宣言の読み口はここにしかない。全量面が
   * `KarumeModel.graph` から得ているものと同一物。
   */
  get graph(): IrGraph {
    return this.#graph;
  }

  /**
   * 必要メモリの見積り（ADR 0070 決定 5）— GPU に触れない純関数で、返るのはカテゴリ別の
   * バイト数と勘定に入っていないものの列だけ。可否の判定はしない（最終門は out-of-memory
   * errorScope のまま）。全量面 `estimateSessionMemory` と**同じ実装 1 本**。
   */
  estimate(options: EstimateOptions = {}): AdmissionReport {
    return estimateGraphMemory(this.#graph, this.#residency, options);
  }

  /**
   * 重み shard 列（グラフ shard を**含まない**）を消費して Session を作る。届いた順に
   * 「進行検証 → CPU 展開（適格外のみ）→ GPU アップロード → フェンス → 参照を手放す」。
   *
   * MUST: 各 shard の bytes は buffer 全体を占める（ADR 0038 §5 と同じ規律 — slice で辻褄を
   * 合わせると RAM ピークが倍増する）。
   * MUST: 途中で失敗したら部分 Session を公開しない（transaction 境界 — {@link Session.build}）。
   * 消費されなかった入力側 iterator も明示的に閉じる（hub の async generator の後始末）。
   */
  async createSession(
    gpu: GpuContext,
    weightShards: AsyncIterable<ModelShard>,
    options: SessionOptions = {},
  ): Promise<Session> {
    const iterator = weightShards[Symbol.asyncIterator]();
    try {
      // MUST: グラフ shard 自身のテンソルも重みとして同じ門を通す（ADR 0070 決定 1 — 検査・
      // アップロード・errorScope 規律を 2 面で分岐させない）。
      const files = followingShards({ file: this.#file, origin: this.#origin }, iterator);
      return await Session.build(gpu, this.#graph, this.#residency, files, options);
    } catch (cause) {
      // MUST: 後始末の失敗で本体の例外を上書きしない（run 側と同じ規律）。
      await iterator.return?.().catch(() => undefined);
      throw cause;
    }
  }
}

/**
 * グラフ shard（配布形の先頭 shard = `__metadata__.karume_ir` + 小テンソル）だけで admission を
 * 済ませる 2 段境界の入口（ADR 0070 決定 5 / graph-first）。
 *
 * 「非対応 op / 格納 dtype」「IR 契約違反」は**ここで**落ちる — 重み shard を 1 バイトも
 * 取得する前に「実行できない」が分かる。続けて {@link PreparedModel.estimate} で必要側の
 * バイト数を、{@link PreparedModel.createSession} で重み shard 列を渡して Session を得る。
 *
 * shard 由来の失敗（parse 不能・グラフ shard でない・IR として壊れている）は `shard [0] 'id'` を
 * 名乗る。capability 不足と契約違反は**帰属を足さない** — モデル（グラフ）の性質であって
 * shard の中身の話ではないので、全量面と同じ文言のままにする。
 */
export const prepareModel = (graphShard: ModelShard): PreparedModel =>
  PreparedModel.fromGraphShard(graphShard);

/**
 * モデルを実行可能な Session にする。重みの GPU アップロードはこの async ステージで行う。
 */
export const createSession = async (
  gpu: GpuContext,
  model: KarumeModel,
  options: SessionOptions = {},
): Promise<Session> =>
  // MUST: 全量面も shard 経路（グラフ shard 1 本 + 重み shard 0 本の列）で構築する — 検査・
  // アップロード・errorScope 規律を 2 面で分岐させない（ADR 0070 受入①の構造的根拠）。
  // 1 shard の列では従来どおり「1 同期区間 + 末尾 submit 1 回 + フェンス」になり、挙動も
  // エラー文言も変わらない。
  // MUST: `async` を外さない — prepare 相（capability 門・契約検査）の失敗は**同期 throw では
  // なく reject** で届ける契約（既存の呼び手は Promise の失敗として捌いている）。
  await PreparedModel.fromModel(model).createSession(gpu, noWeightShards(), options);

/**
 * shard 列（各要素 = 実名 + 独立に整合な safetensors 1 本の bytes = {@link ModelShard}）から
 * Session を作る（ADR 0070 決定 3 の shard 逐次面）。最初の shard はグラフ shard
 * （`__metadata__.karume_ir` 持ち）で、重み shard は届いた順に「検査 → アップロード →
 * フェンス → 参照を手放す」で消費する。同一資産なら {@link createSession}（全量面）と
 * GPU 常駐バイト列・診断が一致する（受入① — 経路自体を共有している）。
 *
 * {@link prepareModel} + {@link PreparedModel.createSession} の薄い合成（列の先頭を自分で
 * 取るだけ）。admission を重み取得の前に挟みたい呼び手は 2 段の面を直接使う。
 *
 * shard 由来の失敗（parse・宣言違反・co-shard・アップロード）は `shard [n] 'id'` を名乗る。
 * hub の `streamAssets` はそのまま渡せる（`StreamedAsset` と構造互換）。
 */
export const createSessionFromShards = async (
  gpu: GpuContext,
  shards: AsyncIterable<ModelShard>,
  options: SessionOptions = {},
): Promise<Session> => {
  const iterator = shards[Symbol.asyncIterator]();
  let prepared: PreparedModel;
  try {
    const first = await iterator.next();
    if (first.done === true) {
      throw new ExecutionError("shard 列が空（最初の shard はグラフ shard — ADR 0070 決定 3）");
    }
    prepared = prepareModel(first.value);
  } catch (cause) {
    // MUST: 後始末の失敗で本体の例外を上書きしない（run 側と同じ規律）。
    await iterator.return?.().catch(() => undefined);
    throw cause;
  }
  // MUST: 残りの shard は**同じ iterator** をそのまま渡す（別の generator で包むと、構築が
  // 失敗したときの `return()` が元の列（hub の async generator）まで届かない）。
  return await prepared.createSession(gpu, { [Symbol.asyncIterator]: () => iterator }, options);
};
