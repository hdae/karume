/**
 * Session の構築相（{@link buildSessionState} = `Session.build` の本体と、構築だけが使う
 * 部品 — ノブの受理集合 / companion scale / 供給単位の取り回し / カナリア /
 * {@link SessionState}）。
 *
 * MUST: executor.ts へ import を張らない（循環 import の禁止 — session-types.ts と同じ規律）。
 * 構築相が返すのは状態（{@link SessionState}）だけで、`Session` は private constructor を
 * 持つため実体はファサード側で作る。依存は {@link "./executor.ts"} → ここの**一方向**。
 */

import { codecLayout } from "../format/container/codecs.ts";
import { alignF16Payload, decodeF16 } from "../format/f16.ts";
import { decodeI2 } from "../format/i2.ts";
import { decodeI4 } from "../format/i4.ts";
import { alignI8Payload, decodeI8 } from "../format/i8.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";
import { RunArena } from "../gpu/arena.ts";
import {
  type AttentionI8a8Decision,
  decideAttentionI8a8Dot,
  formatAttentionI8a8Decision,
} from "../gpu/attention-dp4a-canary.ts";
import { type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import { SessionPipelines } from "../gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../gpu/submit.ts";
import { BUFFER_USAGE } from "../gpu/webgpu-constants.ts";
import { dp4aAvailable } from "../kernels/linear-i8a8.ts";
import type { ScoreStorage } from "../kernels/score-storage.ts";
import type { FusionCounts } from "./fusion.ts";
import { countUses, declaredDtypes, ExecutionError } from "./plan.ts";
import type { GenerationLimits, StepRecipe } from "./recipe.ts";
import type { TransientLimits } from "./transient-plan.ts";
import {
  assertWeightsWithinLimits,
  type ResidentWeight,
  resolveSharedWeights,
  type SharedWeight,
  type WeightResidency,
} from "./weight-residency.ts";
import {
  type ComputePrecision,
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  I8A8_DOT,
  type I8a8Dot,
  type LinearGemvReduce,
  type RmsNormReduce,
  ROW_BLOCK_SPLIT,
  type SessionBuildStats,
  type SessionOptions,
  STATE_ATTENTION_REDUCES,
  type StateAttentionReduce,
  type StorageDiagnostics,
} from "./session-types.ts";

/**
 * {@link SessionOptions} の実行形ノブの受理集合。
 *
 * MUST: 器は `Record<union, true>` — union に値を足してここを直し忘れると、キーの欠落が
 * **型検査で**赤くなる（値の配列で持つと、足した値が黙って受理集合から落ちる）。
 */
const LINEAR_COMPUTES: Readonly<Record<NonNullable<SessionOptions["linearCompute"]>, true>> = {
  f32: true,
  a8: true,
  f16: true,
};
const ATTENTION_COMPUTES: Readonly<Record<ComputePrecision, true>> = {
  f32: true,
  f16: true,
  a8: true,
};
const SCORE_STORAGES: Readonly<Record<ScoreStorage, true>> = { f32: true, f16: true };
const LINEAR_GEMV_REDUCES: Readonly<Record<LinearGemvReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-subgroup32": true,
};

/**
 * 実行形ノブの綴りを {@link "./executor.ts"} の `Session.build` の入口で検査する。
 *
 * MUST: union 外の綴りは fail loudly。下流の消費は全て `=== "a8"` / `=== "f16"` /
 * `=== "parallel"` の等値比較なので、1 文字でも違えば**既定（f32 / sequential）で黙って走る** —
 * opt-in が適用されないまま「a8 を測った」と読める形になる。TS の型で守られているのは TS の
 * 呼び手だけで、JS の呼び手と、改名前の綴りを残したコード（`linearCompute` は 0.5.0 で
 * `"i8a8"` → `"a8"`・ADR 0074 決定 3・互換シムは置かない）は網の外にある。
 * MUST: 違反は**全件列挙して 1 回で落とす**（`assertWeightsWithinLimits` と同じ流儀 — 1 本ずつ
 * 落とすと、直すたびに次の 1 本が現れて何本直せば通るのかが最後まで分からない）。
 * MUST: 受理集合を引く**前に** `typeof` で型（文字列）を見る。`Object.hasOwn` は値をプロパティ
 * キーへ変換するので、この門が無いと `["a8"]` が `'a8'` として受理され、下流の厳密比較では
 * 外れて既定へ黙って縮退する。
 * MUST: 診断でも利用者の変換（`toString` / `Symbol.toPrimitive` / `toJSON`）を呼ばない。非文字列は
 * `typeof` の型名だけを出す — 入力境界の診断で利用者のコードを走らせると、`ExecutionError` の
 * 代わりに利用者側の例外が `createContainerSession` から抜ける。
 *
 * パッケージ内向けに export しているのはテスト用（GPU に触れない純関数なので、アダプタ無し
 * 環境でも回帰を撃てる）。`mod.ts` の公開面には出さない（ADR 0008）。
 */
export const assertExecutionKnobs = (
  linearCompute: NonNullable<SessionOptions["linearCompute"]>,
  attentionCompute: ComputePrecision,
  attentionScoreStorage: ScoreStorage,
  stateAttentionReduce: StateAttentionReduce,
  linearGemvReduce: LinearGemvReduce,
): void => {
  const knobs: readonly (readonly [string, string, Readonly<Record<string, true>>])[] = [
    ["linearCompute", linearCompute, LINEAR_COMPUTES],
    ["attentionCompute", attentionCompute, ATTENTION_COMPUTES],
    ["attentionScoreStorage", attentionScoreStorage, SCORE_STORAGES],
    ["stateAttentionReduce", stateAttentionReduce, STATE_ATTENTION_REDUCES],
    ["linearGemvReduce", linearGemvReduce, LINEAR_GEMV_REDUCES],
  ];
  const violations = knobs
    .filter(([, value, accepted]) => typeof value !== "string" || !Object.hasOwn(accepted, value))
    .map(([name, value, accepted]) =>
      `  - ${name}: ${
        typeof value === "string" ? JSON.stringify(value) : typeof value
      }（受理するのは ${Object.keys(accepted).map((accept) => `'${accept}'`).join(" / ")}）`
    );
  if (violations.length === 0) return;
  throw new ExecutionError(
    `SessionOptions の実行形ノブ ${violations.length} 本が受理集合の外（既定へ黙って縮退させない）:\n` +
      `${violations.join("\n")}\n` +
      "綴りを確認すること（linearCompute の 'i8a8' は 0.5.0 で 'a8' へ改名した — ADR 0074 決定 3）",
  );
};

/** MUST: `queue.writeBuffer` で書くバッファはプール外（アリーナの不変条件）。 */
export const HOST_WRITTEN_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST |
  BUFFER_USAGE.COPY_SRC;

/**
 * companion scale の実体（量子化 codec のみ・piece 列では piece 1 だけ）。
 *
 * `shape` は**rank 2 group 形** `[shape[rowAxis], 行長 / groupSize]`（container-v1 §6.1）。
 * per-channel（i8 / i2）は group 数 1 の `[rows, 1]` で、消費側は 1 形だけを扱う。
 */
export type ReadyScale = {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly shape: readonly [number, number];
};

/**
 * 供給元（コンテナの供給計画）から Session 構築へ渡る initializer 1 本ぶんの実体。
 * **バイト列**で受け渡す（コンテナの block でも呼び手の器でもない — 供給元の形を消費側に
 * 漏らさない）。
 */
export type ReadyInitializer = {
  readonly name: string;
  /** 格納 payload（丸ごと / その piece だけ）。整列の詰め物を含まない生バイト列。 */
  readonly payload: Uint8Array<ArrayBuffer>;
  readonly scale?: ReadyScale;
  /**
   * 分割テンソルの位置（丸ごと 1 本で来たときは undefined — container-v1 §5）。
   *
   * 消費側は `first` でバッファを確保して scale を上げ、`last` でだけ末尾整列の詰め物を掛け、
   * 各 piece を `rowOffset` から決まるバイト位置へ書く（中間 piece に詰め物を掛けると、
   * 詰め物が次の piece の先頭バイトを潰す沈黙誤値になる）。
   */
  readonly piece?: {
    /** この piece が始まる行（先頭次元）。 */
    readonly rowOffset: number;
    /** この piece の行数。 */
    readonly rows: number;
    /** piece 1（バッファ確保と companion scale の転送を担う席）。 */
    readonly first: boolean;
    /** piece n（末尾整列の詰め物を担う席）。 */
    readonly last: boolean;
  };
};

/**
 * 量子化格納の companion scale（ADR 0019 / 0069）。実在・F32・形（rank 2 group 形への正規化）は
 * 供給元（コンテナの合流層）が済ませているので、ここは view を組むだけ。
 *
 * MUST: `Float32Array` の view はコピーせずに張る（scale は重み本体に比べれば小さいが、
 * ここで無条件コピーを挟むと「生バイトのまま常駐」の経路が二重確保になる）。バイト位置の
 * 4 バイト整列は供給元が保証する（safetensors の F32 / コンテナの 64 B 整列 block）。
 */
const scaleTensor = (
  item: ReadyInitializer,
  layout: string,
): {
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly values: Float32Array<ArrayBuffer>;
  readonly shape: readonly [number, number];
} => {
  const scale = item.scale;
  if (scale === undefined) {
    // 存在は型の上でだけ optional なので、黙って読み飛ばさず言い直す（fail loudly）。
    throw new ExecutionError(`initializer '${item.name}': 格納 ${layout} なのに scale が無い`);
  }
  return {
    bytes: scale.bytes,
    values: new Float32Array(
      scale.bytes.buffer,
      scale.bytes.byteOffset,
      scale.bytes.byteLength / 4,
    ),
    shape: scale.shape,
  };
};

/**
 * piece（先頭次元の行範囲）の CPU 展開が読む companion scale の切り出し。
 *
 * scale は rank 2 group 形 `[行, group 数]` で、piece 分割は行の軸が 0 の initializer にしか
 * 許されない（container-v1 §5 規則④ — 供給元が保証）。行の軸が 0 でない形（`[Cin,Cout,K]` の
 * conv_transpose1d）は丸ごとしか来ないので、ここには現れない。
 */
const scaleForPiece = (
  scale: { readonly values: Float32Array<ArrayBuffer>; readonly shape: readonly [number, number] },
  declaredRows: number,
  rowOffset: number,
  rows: number,
): { readonly values: Float32Array<ArrayBuffer>; readonly shape: readonly [number, number] } => {
  if (scale.shape[0] !== declaredRows) return scale;
  const stride = scale.shape[1];
  return {
    values: scale.values.subarray(rowOffset * stride, (rowOffset + rows) * stride),
    shape: [rows, stride],
  };
};

/**
 * per-channel scale（rank 2 `[rows, 1]`）を、CPU 展開（`decodeI8` / `decodeI2` — keepdim broadcast 形の
 * stride で引く）が読む形へ写す。バイト列は同じで、形だけを読み替える。
 */
const keepdimScaleShape = (
  weightShape: readonly number[],
  rows: number,
  rowAxis: number,
): readonly number[] => weightShape.map((_, axis) => (axis === rowAxis ? rows : 1));

/**
 * GPU 常駐経路の per-channel scale が**平坦添字で引ける形**であることを見る（ADR 0019）。
 *
 * カーネルは `wscale[出力チャネル]` と読むので、scale は行の軸だけが伸びた rank 2 `[rows, 1]`
 * （rows = 重みの `shape[rowAxis]`）でなければならない。group 数が 1 でない形（i4 の group 形）は
 * i8 / i2 の席には来ないが、宣言と供給元の食い違いは沈黙誤値になるので**ここが唯一の門**。
 */
const assertRowScale = (
  name: string,
  weightShape: readonly number[],
  scaleShape: readonly [number, number],
  rowAxis: number,
): void => {
  if (scaleShape[0] !== weightShape[rowAxis] || scaleShape[1] !== 1) {
    throw new ExecutionError(
      `initializer '${name}': scale [${scaleShape.join(",")}] が重み [${
        weightShape.join(",")
      }] の軸 ${rowAxis} の per-channel 形 [${weightShape[rowAxis]},1] でない`,
    );
  }
};

/**
 * Session 構築が消費する**供給の単位**（コンテナの part 1 本 = その part にある block の実体列）。
 * フェンス（空 submit + 完了待ち）はこの単位で 1 回、errorScope は `scopePerItem` なら
 * item（block）ごと、そうでなければ batch ごとに張る。
 */
export type WeightBatch = {
  readonly origin: string | undefined;
  readonly items: readonly ReadyInitializer[];
  readonly scopePerItem: boolean;
};

/**
 * 重みアップロード区間のラベル（errorScope とフェンスの帰属先）。`origin` から導出する —
 * batch 側と別々に持つと、片方だけ名乗り方が変わったときに 2 つの名前で同じ失敗が出る。
 */
const uploadLabel = (origin: string | undefined): string =>
  origin === undefined ? "重みのアップロード" : `${origin} の重みアップロード`;

/**
 * 供給単位由来の失敗に帰属先（`part N`）を足して**同じエラーを返す**（`origin` が無ければ
 * 素通し — 文言が 1 文字も変わらない）。
 *
 * MUST: 新しい Error で包み直さない。呼び出し側はクラスで分岐しており（宣言違反 =
 * `ContainerFormatError`）、包むと分岐が壊れて stack も切れる。
 */
const attributeToOrigin = (origin: string | undefined, cause: unknown): unknown => {
  if (origin !== undefined && cause instanceof Error) {
    cause.message = `${origin}: ${cause.message}`;
  }
  return cause;
};

/**
 * 導出相まるごとの成果物（Session 常駐 — キーは {@link "./executor.ts"} の
 * `Session.#preparedKey`）。
 *
 * MUST: 後段が実際に参照するものだけを持つ（`GraphPlan.nodes` / `GraphPlan.bindings` は
 * レシピ導出が終われば誰も読まない）。読まれない導出物を抱えると、キャッシュの寿命が
 * 「run の入出力に効く事実」から離れ、何を再利用しているのかが読めなくなる。
 */
export type PreparedPlan = {
  /** 入力・initializer・全ノード出力の解決済み shape（`#uploadInput` / `#readOutputs` が読む）。 */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly recipes: readonly StepRecipe[];
  /** 計画時に決まった融合回数（ヒット run もこの値を常設診断へ報告する — ADR 0040 §3）。 */
  readonly fusions: FusionCounts;
  /**
   * generation run の run 前検査に要る計画事実（{@link "./recipe.ts"} の
   * `assertGenerationRun`）。state ノードを持たないグラフでは空で、その run は検査を 1 つも
   * 通さない（見る対象が無い）。
   */
  readonly generation: GenerationLimits;
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

export type SessionState = {
  readonly gpu: GpuContext;
  /**
   * 実行するグラフ。MUST: 供給元（開いた容器）を丸ごと持たない — 取得元のバイト列を掴むと
   * 配布ファイル全量の ArrayBuffer が Session の寿命まで固定され、part ごとの逐次消費
   * （ADR 0108 決定 9）の「参照を手放す」契約が成立しない。構築後に要るのは graph だけ。
   */
  readonly graph: IrGraph;
  /**
   * device 寿命のパイプラインキャッシュ（GpuContext 所有）への、この Session ぶんの使用記録つき
   * の面。**キャッシュ自体は Session 常駐ではない** — 同一 device の Session は 1 本を共有する。
   */
  readonly cache: SessionPipelines;
  readonly scheduler: SubmitScheduler;
  /**
   * 中間バッファ計画の上限（device の granted 値 — ADR 0093 決定 1）。計画は Session 構築後に
   * 変わらない値だけを見るので、ここで 1 度固定する。
   */
  readonly transientLimits: TransientLimits;
  readonly weights: RunArena;
  readonly weightBuffers: ReadonlyMap<string, GPUBuffer>;
  /**
   * 重みの常駐分類（prepare 相の純関数の結果 — {@link "./weight-residency.ts"} の
   * `planWeightResidency`）。
   * {@link "./executor.ts"} の `Session.exportWeight` が席を名乗るために構築後も保つ。
   */
  readonly residency: ReadonlyMap<string, WeightResidency>;
  /**
   * この Session が借りている重み（{@link SessionOptions.sharedWeights} の実体）。
   * dispose で借用を返す先で、**貸し手の生存はこの計数が保証する**。
   */
  readonly sharedWeights: readonly SharedWeight[];
  /**
   * params バッファの内容アドレスキャッシュ（キー = usage + 全要素の連結 —
   * `RecipeBuilder.#writeParams`）。実体は weights アリーナが所有する Session 常駐バッファで、
   * ここは「内容 → 既に上げてあるバッファ」の索引だけを持つ。
   * MUST: モジュールスコープに置かない（副作用ゼロの不変条件 — Session ごとに device も
   * バッファも別）。
   */
  readonly paramsCache: Map<string, GPUBuffer>;
  /**
   * 解決済み bindings → 導出済み実行計画（LRU・上限 {@link "./executor.ts"} の
   * `PREPARED_PLAN_CAPACITY`）。
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
  /** slot backing を同時に保持する予算（{@link SessionOptions.planBackingBudgetBytes}）。 */
  readonly planBackingBudgetBytes: number;
  /** 融合 attention の実行形（opt-in — {@link SessionOptions.attentionCompute}）。 */
  readonly attentionCompute: ComputePrecision;
  /** S の格納形（opt-in — {@link SessionOptions.attentionScoreStorage}）。計算形と直交する軸。 */
  readonly attentionScoreStorage: ScoreStorage;
  /**
   * states 形 attention ①QK / ③PV の縮約形（opt-in —
   * {@link SessionOptions.stateAttentionReduce}）。`"parallel"` でも **①' が選ばれるのは M ≤ 8 の
   * 計画だけ**（decode と投機の verify — prefill 計画は ① のまま）で、**③' が選ばれるのは M < 16 の計画だけ**
   * （M ≥ 16 は席に依らず ③ₜ = ③ とビット同一のタイル経路 — 席は 1 つ）。
   */
  readonly stateAttentionReduce: StateAttentionReduce;
  readonly linearGemvReduce: LinearGemvReduce;
  readonly rmsNormReduce: RmsNormReduce;
  /**
   * 行ブロック gemv の並列度目標（opt-in — {@link SessionOptions.linearGemvRowsThreadTarget}）。
   * 省略（`undefined`）はカーネル側の既定 = 参照 device の飽和点。
   */
  readonly linearGemvRowsThreadTarget: number | undefined;
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
  readonly fuseRmsNormAdd: boolean;
  readonly fuseLinearStaticQuantize: boolean;
  /** 固定 SRQ の活性を packed int8 で並列 GEMV へ渡す（ADR 0105）。 */
  readonly packedStaticQuantize: boolean;
  readonly useCounts: ReadonlyMap<string, number>;
  readonly dtypes: ReadonlyMap<string, IrDtype>;
  readonly outputNames: ReadonlySet<string>;
};

/**
 * {@link "./executor.ts"} の `Session.build` の本体（構築トランザクションそのもの）。契約と
 * MUST はファサード側 `Session.build` の doc が正本。
 *
 * 返すのは `new Session(state)` に渡す {@link SessionState} だけ — `Session` は private
 * constructor なので、ここからは実体を作らない。
 */
export const buildSessionState = async (
  gpu: GpuContext,
  graph: IrGraph,
  residency: ReadonlyMap<string, WeightResidency>,
  batches: AsyncIterable<WeightBatch>,
  options: SessionOptions,
): Promise<SessionState> => {
  if (
    options.fuseLinearStaticQuantize !== undefined &&
    typeof options.fuseLinearStaticQuantize !== "boolean"
  ) {
    throw new ExecutionError("options.fuseLinearStaticQuantize はbooleanでなければならない");
  }
  if (options.fuseRmsNormAdd !== undefined && typeof options.fuseRmsNormAdd !== "boolean") {
    throw new ExecutionError("options.fuseRmsNormAdd はbooleanでなければならない");
  }
  if (
    options.packedStaticQuantize !== undefined &&
    typeof options.packedStaticQuantize !== "boolean"
  ) {
    throw new ExecutionError("options.packedStaticQuantize はbooleanでなければならない");
  }
  const rmsNormReduce = options.rmsNormReduce === undefined ? "workgroup" : options.rmsNormReduce;
  if (rmsNormReduce !== "workgroup" && rmsNormReduce !== "subgroup32") {
    // 診断で利用者の変換を呼ばない（`String(x)` は `Symbol.toPrimitive` / `toString` を走らせ、
    // 例外を投げるオブジェクトでは `ExecutionError` の代わりにそれが抜ける）。
    throw new ExecutionError(
      `options.rmsNormReduce: 未対応の値 ${
        typeof rmsNormReduce === "string" ? JSON.stringify(rmsNormReduce) : typeof rmsNormReduce
      }`,
    );
  }
  if (
    rmsNormReduce === "subgroup32" &&
    (!gpu.features.has("subgroups") || !gpu.features.has("subgroup-size-control") ||
      !gpu.wgslLanguageFeatures.has("subgroup_id"))
  ) {
    throw new ExecutionError(
      "rmsNormReduce: subgroup32 は acquireGpu({ subgroups: true }) が必要",
    );
  }
  const linearCompute = options.linearCompute ?? "f32";
  const attentionCompute = options.attentionCompute ?? "f32";
  const attentionScoreStorage = options.attentionScoreStorage ?? "f32";
  const stateAttentionReduce = options.stateAttentionReduce ?? "sequential";
  const linearGemvReduce = options.linearGemvReduce ?? "sequential";
  const planBackingBudgetBytes = options.planBackingBudgetBytes ??
    DEFAULT_PLAN_BACKING_BUDGET_BYTES;
  // MUST: 綴りの検査は既定代入の直後・以降の全ゲートより前。ここを通った後は s16×c16 ゲートも
  // f16 feature ゲートも union 内の値だけを見ればよい。
  assertExecutionKnobs(
    linearCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
  );
  if (
    options.fuseLinearStaticQuantize === true &&
    (linearGemvReduce !== "parallel" || linearCompute !== "f32")
  ) {
    throw new ExecutionError(
      "fuseLinearStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応",
    );
  }
  // packed 活性の変種を持つのは並列 GEMV 族だけ（ADR 0105）。黙って f32 経路へ落とすと
  // 「指定したのに効かない」席になるので、融合と同じ流儀で拒否する。
  if (
    options.packedStaticQuantize === true &&
    (linearGemvReduce !== "parallel" || linearCompute !== "f32")
  ) {
    throw new ExecutionError(
      "packedStaticQuantize は linearGemvReduce: parallel / linearCompute: f32 のみ対応",
    );
  }
  if (linearGemvReduce !== "sequential" && linearCompute !== "f32") {
    throw new ExecutionError(
      `linearGemvReduce: ${linearGemvReduce} は linearCompute: f32 のみ対応`,
    );
  }
  if (
    linearGemvReduce === "parallel-subgroup32" &&
    (!gpu.features.has("subgroups") || !gpu.features.has("subgroup-size-control") ||
      !gpu.wgslLanguageFeatures.has("subgroup_id"))
  ) {
    throw new ExecutionError(
      "linearGemvReduce: parallel-subgroup32 は acquireGpu({ subgroups: true }) が必要",
    );
  }
  // 値域の検査（union を読まない）は綴りの門の後 — 文言は estimate.ts の同じ門と揃える。
  if (!Number.isSafeInteger(planBackingBudgetBytes) || planBackingBudgetBytes < 0) {
    throw new ExecutionError(
      `options.planBackingBudgetBytes ${String(planBackingBudgetBytes)} は非負の安全な整数で` +
        "なければならない",
    );
  }
  // 行ブロック gemv の並列度目標も同じ形で検査する（0 / 負 / 非整数は「スレッド数の目標」として
  // 意味を持たず、黙って受けると rows が最大のまま選ばれた走行になる）。
  const linearGemvRowsThreadTarget = options.linearGemvRowsThreadTarget;
  if (
    linearGemvRowsThreadTarget !== undefined &&
    (!Number.isSafeInteger(linearGemvRowsThreadTarget) || linearGemvRowsThreadTarget < 1)
  ) {
    throw new ExecutionError(
      `options.linearGemvRowsThreadTarget ${String(linearGemvRowsThreadTarget)} は 1 以上の` +
        "安全な整数でなければならない",
    );
  }
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

  // MUST: 重みの確保に入る前に、席ごとの確保寸法を device の絶対上限と突き合わせる（batch
  // ループより前 = 1 バイトも上げる前）。確保失敗の検出は item（block）単位 errorScope
  // （ADR 0108 決定 9）が担うが、それは実装の報告品質に依存し（out-of-memory scope が黙る
  // device が実在する — docs/known-issues.md の Metal 節）、捕まえても数 GiB
  // 転送した後にしか出ない。寸法は宣言だけで確定している（常駐計画は prepare 相の純関数）ので、
  // 決定論的に落とせるぶんはここで落とす（同 known-issues が名指しした「明示サイズ門」）。
  // NOTE: 見るのは絶対上限だけで空き VRAM とは比べない（ADR 0070 決定 5 の規律 — 検査は
  // 純粋な比較のままで、総量の可否の最終門は errorScope に残る）。
  assertWeightsWithinLimits(residency, gpu.limits);

  // 共有 initializer（借り物の重み — ADR 0096 段 2 §1.3）の突合。**バイトを 1 つも上げる前**に
  // 席・宣言 shape・貸し手の codec・device を見る（門の中身は `resolveSharedWeights`）。
  const shared = resolveSharedWeights(graph, residency, gpu, options.sharedWeights);

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
  /**
   * 展開席の piece 列が持ち越す companion scale の**写し**（キー = initializer 名）。
   *
   * MUST: view ではなく値の写しを持つ。scale の実体は piece 1 と同じ part にしか無く
   * （規則③）、view のまま抱えるとその part のバイト列が列の最後まで解放されず、
   * RAM ピーク O(最大 part) が崩れる。写すのは scale だけで、重み本体は 1 バイトも写さない。
   */
  const carriedScales = new Map<
    string,
    { readonly values: Float32Array<ArrayBuffer>; readonly shape: readonly [number, number] }
  >();
  let residentCompressedBytes = 0;
  let hostExpandedBytes = 0;
  // 構築相の費用内訳（{@link SessionBuildStats}）。ホスト時計だけで刻む集計器で、
  // MUST NOT: 計測のために GPU フェンスを足さない・submit の位置を動かさない
  // （batch ごと submit 1 回という ADR 0108 決定 9 の契約が崩れると、瞬間ピークが重み 1 本ぶん
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
  // 書き込み先オフセットは piece 列（分割テンソル）のためにある — 丸ごとの経路は常に 0 で、
  // piece は「行オフセット × 1 行のバイト長」を渡して 1 本のバッファへ継ぎ足す。
  const timedWrite = (
    buffer: GPUBuffer,
    data: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer>,
    offset: number,
  ): void => {
    const start = performance.now();
    gpu.device.queue.writeBuffer(buffer, offset, data);
    writeBufferIssueMs += performance.now() - start;
    uploadedBytes += data.byteLength;
  };
  /** 借用を積み終えた共有 initializer（構築が失敗したらここから 1 本ずつ返す）。 */
  const borrowed: SharedWeight[] = [];
  try {
    // 借り物の重みは block を 1 つも読まずに台帳へ載る（バイトは貸し手が既に GPU へ
    // 上げている）。借用計数を先に積むのは、構築中に貸し手が dispose される窓を塞ぐため。
    for (const { name, shared: weight } of shared) {
      const internals = weight[RUNTIME_INTERNAL];
      internals.retain();
      borrowed.push(weight);
      weightBuffers.set(name, internals.buffer);
      // 圧縮席（f16 / i8 / i4）は貸し手の付随実体（scale・group 長）ごと引き継ぐ。ここに
      // 載らない名前は f32 として読まれる（重み台帳の既定）ので、席の突合が門になっている。
      if (internals.resident !== undefined) residentWeights.set(name, internals.resident);
    }
    // batch の反復待ち（= 供給側の費用）は for await が隠すので、**前の batch を処理し終えた
    // 時刻**との差で測る（次の batch が届くまでの間はこの 2 点の間にしか無い）。
    let shardBoundary = performance.now();
    for await (const batch of batches) {
      shardWaitMs += performance.now() - shardBoundary;
      shardCount += 1;
      // errorScope とフェンスは同じラベルを名乗る MUST（別々に組むと同じアップロード区間の
      // 失敗が 2 つの名前で出る）。
      const label = uploadLabel(batch.origin);
      /** initializer 1 本ぶん（丸ごと / 1 piece）の展開とアップロード。同期区間の中で呼ぶ。 */
      const uploadItem = (item: ReadyInitializer): void => {
        const name = item.name;
        const initializer = graph.initializers[name];
        const raw = item.payload;
        // 席はプランナが正本（全 initializer を載せる契約 — 欠けは簿記の破れ）。
        const seat = residency.get(name);
        if (seat === undefined) {
          throw new ExecutionError(`initializer '${name}': 常駐分類が無い`);
        }
        // MUST: 借り物の席に実体が来る形は落とす。共有 initializer は突合集合の外
        // （供給元）なので `items` には現れない — 現れたら簿記の破れで、通すと貸し手の
        // バッファを指す名前に別のバイト列を上書きすることになる。
        if (seat.seat === "shared" || initializer.storage === undefined) {
          throw new ExecutionError(
            `initializer '${name}': 共有宣言（shared）なのに実体が来た`,
          );
        }
        const storage = initializer.storage;
        const layout = codecLayout(storage.codec);
        // initializer の宣言 shape は数値のみ（パーサが保証 — 記号次元は拒否）。
        const declaredShape = graph.values[name].shape.map(Number);
        const declaredRows = declaredShape[0];
        // 分割テンソル（piece 列）は「先頭次元の連続範囲」で届く。展開に渡す shape はその
        // piece の形、バイト位置と長さは**宣言由来の 1 行あたりバイト長**の按分で決まる
        // （行あたりの長さは宣言から割り切れる — 供給元が shape の残り次元を突き合わせて
        // いるので、行数だけが piece ごとに変わる）。
        const piece = item.piece;
        const pieceShape = piece === undefined
          ? declaredShape
          : [piece.rows, ...declaredShape.slice(1)];
        const rows = pieceShape[0];
        // MUST: 宣言由来のバイト長と現物が食い違ったら落とす。プランナ（と見積り）は実
        // テンソルを見ずに宣言だけで数えるので、ここが「宣言 = 現物」を実際に確かめる唯一の
        // 点になる（供給元の突合門が成立していれば発火しない — 二重の網）。
        const expectedBytes = piece === undefined
          ? seat.payloadBytes
          : rows * (seat.payloadBytes / declaredRows);
        if (raw.byteLength !== expectedBytes) {
          throw new ExecutionError(
            `initializer '${name}': 宣言由来 ${expectedBytes} バイトに対し実テンソルが ${raw.byteLength} バイト`,
          );
        }
        // 書き込み先のバイト位置（丸ごとは常に 0）。生バイト席は格納バイト列、展開席は f32
        // 展開後のバイト列が GPU に載るので、按分の基準になる全体長が席で違う。
        const wholeBytes = seat.seat === "expanded" ? seat.expandedBytes : seat.payloadBytes;
        const byteOffset = piece === undefined ? 0 : piece.rowOffset * (wholeBytes / declaredRows);
        // 末尾のゼロ詰めを掛けてよいのは「丸ごと」と「piece 列の末尾」だけ。中間 piece に
        // 掛けると詰め物が次の piece の先頭バイトを 0 で潰す（中間 piece が 4 バイト整列で
        // あることは供給元の担当 — こちらは詰め物を掛けない側で不変条件を守る）。
        const tailAligned = piece === undefined || piece.last;
        /**
         * 展開席（CPU で f32 化）が読む scale（rank 2 group 形）— piece 列ではその piece の行範囲
         * だけを返す。実体は piece 1 にしか無いので、そこで値を写して列の最後まで持ち越す
         * （{@link carriedScales} の MUST）。
         */
        const expandedScale = (): {
          readonly values: Float32Array<ArrayBuffer>;
          readonly shape: readonly [number, number];
        } => {
          if (piece === undefined) return scaleTensor(item, layout);
          if (piece.first) {
            const scale = scaleTensor(item, layout);
            carriedScales.set(name, { values: new Float32Array(scale.values), shape: scale.shape });
          }
          const carried = carriedScales.get(name);
          if (carried === undefined) {
            throw new ExecutionError(
              `initializer '${name}': piece の scale が piece 1 から持ち越されていない`,
            );
          }
          return scaleForPiece(carried, declaredRows, piece.rowOffset, rows);
        };
        // 格納 f16 / i8 / i4 / i2 だけが 2 経路に分かれる（ADR 0018 / 0019 / 0069 / 0097）。適格なら
        // 生バイトのまま常駐させ dequant はカーネル内（VRAM 削減はこれで初めて成立する）、
        // 適格外はここで f32 へ展開する（正しさは保たれ VRAM 削減はゼロ）。他の格納は
        // 生バイトがそのまま GPU 表現。
        let payload: Uint8Array<ArrayBuffer> | Float32Array<ArrayBuffer> = raw;
        if (layout === "f16") {
          if (seat.seat === "f16") {
            // MUST: 奇数要素長は末尾 2 バイトのゼロ詰めで 4 バイト整列させる。writeBuffer は
            // 4 の倍数でないサイズを validation で拒む（= 重みが空のまま走り出す）。
            payload = tailAligned ? alignF16Payload(raw) : raw;
            residentWeights.set(name, { storage: "f16" });
            residentCompressedBytes += payload.byteLength;
          } else {
            payload = timedDecode(() => decodeF16(raw));
            hostExpandedBytes += payload.byteLength;
          }
        }
        if (layout === "i8" || layout === "i2") {
          const rowAxis = storage.rowAxis ?? 0;
          if (seat.seat === "i8" || seat.seat === "i2") {
            // scale は分割前の**全体**に掛かる 1 本きりなので、形の突合も確保も転送も
            // piece 1（丸ごとなら唯一の実体）でだけ行う。突合に渡すのは piece の形では
            // なく宣言 shape。
            if (piece === undefined || piece.first) {
              const scale = scaleTensor(item, layout);
              assertRowScale(name, declaredShape, scale.shape, seat.rowAxis);
              // MUST: scale のバッファも「GPU 常駐圧縮」に数える（実際に抱えるバイト数）。
              residentCompressedBytes += scale.bytes.byteLength;
              const scaleBuffer = timedAlloc(Math.max(4, scale.bytes.byteLength));
              if (scale.bytes.byteLength > 0) {
                timedWrite(scaleBuffer, scale.bytes, 0);
              }
              residentWeights.set(name, { storage: layout, scale: scaleBuffer });
            }
            // MUST: 要素数が 4 の倍数でない重みは末尾をゼロ詰めして 4 バイト整列させる
            // （f16 の 2 バイト詰めと同じ理由 — writeBuffer が validation で落ちる）。
            payload = tailAligned ? alignI8Payload(raw) : raw;
            residentCompressedBytes += payload.byteLength;
          } else {
            const scale = expandedScale();
            const scaleShape = keepdimScaleShape(pieceShape, scale.shape[0], rowAxis);
            payload = timedDecode(() =>
              layout === "i2"
                ? decodeI2(raw, pieceShape, scale.values, scaleShape)
                : decodeI8(raw, pieceShape, scale.values, scaleShape)
            );
            hostExpandedBytes += payload.byteLength;
          }
        }
        if (layout === "i4") {
          // 適格は f16 / i8 より狭い「消費が linear / embedding / conv1d(groups==1) の
          // 重みスロットのみ」（ADR 0069 決定 5 とその追補 — 展開経路が GEMM 骨格のタイル
          // 読み〈linear は B 側・conv1d igemm は A 側〉と embedding のカーネルにしか無い）。
          // 展開経路の無い重みスロット（conv2d / conv_transpose1d / groups > 1 の conv1d）と
          // 共有される i4 は CPU 展開の受け皿へ（正しさは保たれ VRAM 削減はゼロ —
          // i8 の適格外と同じ設計）。判定はプランナが済ませている。
          if (seat.seat === "i4") {
            // ペイロードは詰め物不要で常に 4 バイト整列 — バイト長 = numel / 2 で、numel は
            // group_size（2 冪 ≥ 16）の倍数だからバイト長は 8 の倍数（ADR 0069 決定 2）。
            // piece の行あたり長も同じ理由で 8 の倍数になる。
            residentCompressedBytes += payload.byteLength;
            if (piece === undefined || piece.first) {
              const scale = scaleTensor(item, layout);
              // MUST: scale のバッファも「GPU 常駐圧縮」に数える（i8 と同じ — 実際に抱える
              // バイト数。exporter の storage_breakdown と診断の意味を揃える）。
              residentCompressedBytes += scale.bytes.byteLength;
              const scaleBuffer = timedAlloc(Math.max(4, scale.bytes.byteLength));
              if (scale.bytes.byteLength > 0) {
                timedWrite(scaleBuffer, scale.bytes, 0);
              }
              // group 長は宣言から写した 1 箇所（プランナ）だけが決める — 別経路で渡せる形に
              // すると「group 64 の資産が group 32 のパイプラインで走る」沈黙誤値になる。
              residentWeights.set(name, {
                storage: "i4",
                scale: scaleBuffer,
                groupSize: seat.groupSize,
              });
            }
          } else {
            // 値域（2 冪 ≥ 16・整除）は合流層 / 旧パーサが保証済み。存在は型の上でだけ optional
            // なので、黙って読み飛ばさず言い直す（「格納 i8 なのに scale が無い」と同じ流儀）。
            const groupSize = storage.groupSize;
            if (groupSize === undefined) {
              throw new ExecutionError(`initializer '${name}': 格納 i4 なのに groupSize が無い`);
            }
            const scale = expandedScale();
            payload = timedDecode(() =>
              decodeI4(raw, pieceShape, scale.values, scale.shape, groupSize)
            );
            hostExpandedBytes += payload.byteLength;
          }
        }
        // バッファの確保は丸ごと 1 回 / piece 列なら先頭 1 回。piece でも寸法は**全体ぶん**
        // を宣言から出す（分割は GPU 側の配置を 1 バイトも変えない — 生バイト席は格納
        // バイト長の 4 バイト切り上げ = 末尾詰め物ぶん、展開席は f32 展開後のバイト長）。
        if (piece === undefined) {
          weightBuffers.set(name, timedAlloc(Math.max(4, payload.byteLength)));
        } else if (piece.first) {
          const aligned = seat.seat === "expanded"
            ? seat.expandedBytes
            : seat.payloadBytes + ((4 - (seat.payloadBytes % 4)) % 4);
          weightBuffers.set(name, timedAlloc(Math.max(4, aligned)));
        }
        const buffer = weightBuffers.get(name);
        if (buffer === undefined) {
          throw new ExecutionError(`initializer '${name}': piece 1 で確保したバッファが台帳に無い`);
        }
        if (payload.byteLength > 0) timedWrite(buffer, payload, byteOffset);
        // 持ち越した scale は列を読み切ったところで捨てる（生きているのは 1 列ぶんだけ）。
        if (piece?.last === true) carriedScales.delete(name);
      };
      // MUST: 重みアップロードも errorScope で囲む（ADR 0004 の「errorScope 常設」）。上限超過の
      // createBuffer は同期例外を投げずに無効バッファを返し、無効バッファ / 整列違反への
      // writeBuffer も警告すら出さない no-op になるため、包まないと重みが空のまま走り出す。
      // MUST NOT: この区間の中で await しない。push から pop の発行までを 1 つの同期区間に
      // 保つことが、device 単位ロックを取らずに LIFO の交錯を防いでいる根拠になっている。
      // 区間の粒度: block（item）ごと（ADR 0108 決定 9 — push / pop は 1.81 µs / 回でほぼ無料。
      // 費用の主はフェンスなのでフェンスは batch = part ごと 1 回に留める）。
      const groups = batch.scopePerItem ? batch.items.map((item) => [item]) : [batch.items];
      for (const group of groups) {
        pushFailureScopes(gpu.device);
        try {
          for (const item of group) uploadItem(item);
        } catch (cause) {
          // MUST: push した 2 本は必ず pop して積み残さない（積み残すと以後の検証結果が誤った
          // スコープに吸われ、エラーが恒久的に見えなくなる）。破棄は外側の transaction 境界が
          // 1 箇所で持つ。
          await discardFailureScopes(gpu.device);
          throw attributeToOrigin(batch.origin, cause);
        }
        const failure = await popFailureScopes(
          gpu.device,
          batch.scopePerItem ? `${label}（initializer '${group[0].name}'）` : label,
        );
        if (failure !== undefined) throw failure;
      }

      // MUST: batch（コンテナの part）ごとに**実際の submit を 1 回**出して完了まで待つ
      // （ADR 0108 決定 9）。queue.writeBuffer は staging を確保して溜め込み、submit の完了まで
      // それを解放しない — 数 GiB の重みを上げた直後は VRAM が二重計上のまま最初の run に入り、
      // 初回ピークが重み 1 本ぶん押し上がる（f16 preset で実測 +2.7GiB。
      // docs/research/2026-08-08-vram-oom-misreport.md §4）。逐次消費ではこの解放が
      // RAM ピーク O(最大 batch) の成立条件そのものになる。フェンスの後にループ末尾へ抜けて
      // batch への参照が尽きる — CPU 側バイト列は転送完了後にだけ手放される
      // （フェンス後解放の順序契約 — ADR 0108 決定 9）。
      // MUST NOT: scheduler.flush() で代用しない。pending dispatch が空だと submit を出さずに
      // 即 return するため、staging は溜まったまま残る。
      // NOTE: submit ごとの onSubmittedWorkDone を禁じているのは run のホットパス（submit.ts の
      // 「計測の帰属」）で、ここは batch ごと 1 回・窓の外なので推定にも壁時計にも乗らない。
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
  } catch (cause) {
    // transaction 境界（ADR 0108 決定 9）: 途中の batch で失敗したら（宣言違反・入力列の例外・
    // GPU エラーのいずれでも）、アップロード済みの重みごと weights アリーナを破棄して
    // 部分 Session を公開しない。
    // MUST: 後始末の失敗で本体の例外を上書きしない（run 側と同じ規律）。原因は本体側に
    // あり、destroy の rejection（主因は device 消失）に差し替わると調査の起点が消える。
    await weights.destroy().catch(() => undefined);
    // MUST: 積んだ借用は必ず返す（返し損ねると貸し手 Session が永久に dispose できない）。
    for (const weight of borrowed) weight[RUNTIME_INTERNAL].release();
    throw cause;
  }

  return {
    gpu,
    graph,
    // MUST: パイプラインキャッシュは GpuContext 所有の 1 本を借りる（device 寿命 —
    // `GpuContextInternals.pipelines`）。ここで新しく割ると、同一 device の Session ごとに
    // 同じ WGSL のコンパイルと getBindGroupLayout の解決を払い直す。
    // MUST: 借りるのは**構築の決着点だけ**で、構築相の途中では 1 本も引かない。構築相は
    // GpuContext のスコープロックの外なので、ここでパイプラインを生成すると並行構築の
    // errorScope が誤帰属する（gpu/device.ts「errorScope 区間の不変条件」）。実際の生成は
    // 全て初回 run のミス経路（RecipeBuilder）= ロックの内側で起きる。
    cache: new SessionPipelines(gpu[RUNTIME_INTERNAL].pipelines()),
    scheduler,
    transientLimits: {
      maxBufferSize: gpu.limits.maxBufferSize,
      maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
      offsetAlignment: gpu.device.limits.minStorageBufferOffsetAlignment,
    },
    weights,
    weightBuffers,
    residency,
    sharedWeights: borrowed,
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
    stateAttentionReduce,
    linearGemvReduce,
    linearGemvRowsThreadTarget,
    planBackingBudgetBytes,
    // linear の拡張の有無は**速度にしか効かない**（両変種は同じ整数を返す）ので、機能検出では
    // なく経路選択としてここで 1 度だけ決める（src/kernels/linear-i8a8.ts の docstring）。
    linearI8a8Dot: options[I8A8_DOT] ?? (dp4a ? "dp4a" : "emu"),
    // attention は同じ主張が実機で反証されている（Metal / Apple M2）ので、列挙ではなく
    // **実走カナリアの判定**（上の `attentionI8a8Dot`）で決める。
    attentionI8a8Dot,
    rowBlockSplit: options[ROW_BLOCK_SPLIT],
    fuseRmsNormAdd: options.fuseRmsNormAdd ?? false,
    fuseLinearStaticQuantize: options.fuseLinearStaticQuantize ?? false,
    packedStaticQuantize: options.packedStaticQuantize ?? false,
    rmsNormReduce,
    useCounts: countUses(graph),
    dtypes: declaredDtypes(graph),
    outputNames: new Set(graph.outputs),
  };
};
