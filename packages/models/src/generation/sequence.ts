/**
 * 1 会話ぶんの寿命を持つ生成の実体（ADR 0083 決定 1〜5 の `GenerationSequence`）。
 * **パイプライン非依存の共通処理**なので `program.ts` / `sampler.ts` と同じ `src/generation/` に置く。
 *
 * ## 可変状態は 2 つだけ
 *
 * MUST: sequence が持つ可変状態は **`context` と `pendingToken` の 2 つだけ**である（ADR 0083
 * 決定 1）。position / totalLength といった counter は持たず、run を組む直前に
 * `context.pastLength` を読む — 論理長の進行は run の成功で context が進める（ADR 0066 決定 6 の
 * **二重簿記の禁止**）。イベントに載せる `position` も、その run の**後**の `context.pastLength`
 * をその場で読んだ値で、保存しない。
 *
 * ## `pendingToken` — 最大 1 token の未 commit frontier
 *
 * `GenerationContext` は常に最大 1 token の未 commit frontier を持つ（K token 生成後の
 * `pastLength = T + K − 1`。decode は `maxNewTokens − 1` 回しか回らないため）。よって次ターンは
 * **`pendingToken` を新しい prompt の先頭に連結して prefill する**（ADR 0083 決定 4）。連結を
 * 落とすと「直前 assistant の最後の token が履歴から 1 個消える」— 例外にならない沈黙劣化で、
 * この経路の門が段 3 の合格線である。
 *
 * MUST: `pendingToken` は**選んだ直後に**更新する（yield の前）。`break` / `return()` は
 * `finally` へ入るだけで、そこから「どこまで進んだか」を再構成する術は無い。選んだ瞬間に
 * 書いておけば、中断がどの yield で起きても値は正しい。
 *
 * MUST: `rewind` は使わない — sliding スロットを含む context は全拒否（ADR 0066 追記 2）。
 * 編集・分岐は「新しい context + token transcript の replay」が正。
 *
 * ## 中断は「完了した run のぶんだけ進んだ状態」で閉じる
 *
 * `break` / `return()` / `AbortSignal` のどれで閉じても、会話は**成功した run のぶんだけ**進んで
 * いる。token を 1 つも受け取っていない中断（= prefill の途中）は prompt が途中まで会話へ入った
 * 状態で、`prefill` イベントの `chunk` が commit 済み chunk 数を表す。続きを送るか sequence を
 * 捨てるかは呼び手が決める（会話の管理はホストの責務 — 決定 10 と同じ線）。
 *
 * ## cancel は `AbortSignal` が正
 *
 * 段の境目（各 run の直前）で検査し、`signal.reason` を**包まずそのまま** throw する
 * （ADR 0083 決定 5 — 前例は `AnimaPipelineOptions.signal`）。
 */

import type {
  GenerationContext,
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  SymbolBindings,
  Tensor,
} from "@karume/runtime";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { planPrefillChunks } from "./greedy.ts";
import { createSampler, isStopToken, type SamplerSpec } from "./sampler.ts";
import type { GenerationWiring } from "./program.ts";

/** {@link GenerationCapacityError} が踏んだ上限（どちらも「もう入らない」）。 */
export type GenerationCapacityConstraint = "capacity" | "maxPosition";

/**
 * {@link GenerationCapacityError} が運ぶ実値（`assertBudget` が落とす**その時点**の観測）。
 *
 * MUST: 全部その場で導出した値である（保存した counter から作らない = 二重簿記の禁止）。
 * MUST: 文言を読み解かせない。切り詰めの判断に要る数を欄で渡すのが専用型を持つ意味である。
 */
export type GenerationCapacityDetail = {
  /** 踏んだ上限（`capacity` = state 容量 / `maxPosition` = 位置表の行数）。 */
  readonly constraint: GenerationCapacityConstraint;
  /** 検査した時点の会話の論理長（`GenerationSequence.used` から未 commit frontier を除いた値）。 */
  readonly pastLength: number;
  /**
   * このターンが流す token 数 — **`pendingToken` の連結後**である。
   *
   * 呼び手が渡した `prompt` より 1 多いことがある（前ターンが token を出していれば連結される）。
   * 切り詰めの計算はこちらの値で行う（次ターンでも同じ 1 token が先頭に付く）。
   */
  readonly promptLength: number;
  /** 落ちた要求の `maxNewTokens`。 */
  readonly requestedNewTokens: number;
  /** `constraint` が指す上限の値。 */
  readonly limit: number;
  /**
   * この `promptLength` のまま**今なら**通る `maxNewTokens` の上限。
   *
   * 0 以下なら prompt だけで入らない（負値は「何 token 溢れているか」を保つ — 切り詰めの目安）。
   * 両制約とも `K ≤ limit − pastLength − promptLength + 1` に畳める（`capacity` は包括上限・
   * `maxPosition` は排他上限で、`assertBudget` の 2 本の式がちょうど同じ形になる）。
   */
  readonly maxNewTokens: number;
};

/**
 * 容量を超えた（= この会話はもう入り切らない）— ADR 0083 決定 10。
 *
 * MUST: 専用の型で落とす。ランタイムも `pastLength + queryLength ≤ 容量` を拒否するが、それは
 * run のエンコード直前の汎用メッセージで、呼び手は「切り詰めれば通る」のか「配線が壊れている」
 * のかを文言から読み分けることになる。**会話の切り詰めはホストの責務**（limitations）なので、
 * その判断に要る 1 件だけを型で分ける。
 *
 * 位置表の上限（`maxPosition`）超過も同じ型で落とす — 呼び手にとっては同じ「もう入らない」で、
 * 打つ手（古い turn を落とす / 新しい context を作る）も同じである。どちらを踏んだかと、
 * そこから切り詰めを計算するのに要る実値は {@link GenerationCapacityDetail} の欄が運ぶ。
 */
export class GenerationCapacityError extends Error {
  readonly constraint: GenerationCapacityConstraint;
  readonly pastLength: number;
  readonly promptLength: number;
  readonly requestedNewTokens: number;
  readonly limit: number;
  readonly maxNewTokens: number;

  constructor(message: string, detail: GenerationCapacityDetail) {
    super(message);
    this.name = "GenerationCapacityError";
    this.constraint = detail.constraint;
    this.pastLength = detail.pastLength;
    this.promptLength = detail.promptLength;
    this.requestedNewTokens = detail.requestedNewTokens;
    this.limit = detail.limit;
    this.maxNewTokens = detail.maxNewTokens;
  }
}

/** 生成中のイベント（ADR 0083 決定 2）。 */
export type GenerationEvent =
  | {
    readonly kind: "token";
    /** 選ばれた token id。 */
    readonly id: number;
    /** この token が会話に置かれる絶対位置（= 直後の `context.pastLength`）。 */
    readonly position: number;
  }
  | {
    readonly kind: "prefill";
    /** **commit 済み**の prefill chunk 数（1 始まり — `chunk / chunks` がそのまま進捗）。 */
    readonly chunk: number;
    readonly chunks: number;
  };

/**
 * 停止理由（{@link GenerationStream.done} が返す）。
 *
 * 停止 token で閉じた枝は 2 つあり、どちらも**停止 token 自体**を運ぶ — `eos` は配布形が宣言した
 * 集合（`GenerationWiring.stopTokens`）で、`stop-token` はその要求だけが足した集合
 * （{@link GenerationRequest.stopTokens}）である。両方に居る id は `eos` で閉じる（配布形の
 * 終端記号としての意味が優先する）。停止 token は `token` イベントに出さない（本文ではなく
 * 終端記号で、chat では `<turn|>` のような書式トークンになる）が、会話には残る = 次ターンの
 * prefill 先頭へ連結される `pendingToken` である。
 *
 * `closed` は消費側が `break` / `return()` で閉じた場合（`aborted` は `AbortSignal` 経由）。
 */
export type GenerationStop =
  & {
    /**
     * このターンが**生成した** token の数（prompt は含まない）。
     *
     * MUST: 停止 token（`eos` / `stop-token`）も 1 個として数える。抽選 1 回 = run 1 回なので、
     * この数がそのまま生成に費やした run 数と一致し、`tok/s` を再エンコード無しで書ける
     * （それがこの欄の目的）。本文だけの数（= `token` イベントの数）が要るなら、停止 token を
     * 運ぶ 2 枝（`"eos"` / `"stop-token"`）のとき 1 引く。
     *
     * `max-tokens` なら要求の `maxNewTokens` に一致し、`closed` / `aborted` では打ち切りまでに
     * 出した数になる（どちらも「成功した run のぶんだけ会話は進んでいる」— 上の節と同じ線）。
     */
    readonly tokens: number;
  }
  & (
    | { readonly reason: "eos"; readonly token: number }
    | { readonly reason: "stop-token"; readonly token: number }
    | { readonly reason: "max-tokens" }
    | { readonly reason: "aborted" }
    | { readonly reason: "closed" }
  );

/**
 * 1 回ぶんの生成リクエスト（ADR 0083 決定 1）。
 *
 * MUST: 中身は {@link GenerationSequence.generate} が**発行時に写す**（ADR 0083 追記
 * 2026-09-02）。発行後にこの object や `prompt` / `stopTokens` 配列・`sampler` の指定を
 * 書き換えても、走行中の生成には効かない — 次のターンに効かせたいなら次の要求として渡す。
 */
export type GenerationRequest = {
  /**
   * 今ターンぶんの token 列。多ターンでは**新しい turn のぶんだけ**を渡す（過去は context の
   * KV にある）。前ターンが token を出していれば `pendingToken` が先頭へ連結されるので、
   * 「続きを生成するだけ」のターンは空配列でよい。
   */
  readonly prompt: readonly number[];
  /** 生成する token 数の上限（1 以上）。停止 token はこの数に**含めない**。 */
  readonly maxNewTokens: number;
  /**
   * このターンだけ効かせる**追加の**停止 token（配布形が宣言した集合との**和集合**で判定する）。
   *
   * 配布形の EOS（`GenerationWiring.stopTokens`）は常に効くので、ここに書くのは「この要求の
   * 都合で止めたい id」だけである（chat の書式トークンで止めたい・道具呼び出しの開始札で
   * 切りたい、など）。止まったことは {@link GenerationStop} の `stop-token` で読み分けられる。
   *
   * MUST: 語彙外と重複は fail loudly（黙って無視すると「効かない停止条件」が例外なしで残り、
   * 出力が伸び続けることでしか気づけない）。
   */
  readonly stopTokens?: readonly number[];
  /** sampling の指定（省略時は温度 0 = greedy — ADR 0083 決定 7 のこの層の既定）。 */
  readonly sampler?: SamplerSpec;
  /** 中断（段の境目で検査し `signal.reason` をそのまま throw する）。 */
  readonly signal?: AbortSignal;
};

/**
 * token 列そのもの（`for await` で汲む）+ 停止理由。
 *
 * MUST: `done` は**二次的な**通知路である。失敗（run の失敗・容量超過）は iterable 側が throw
 * するのが一次で、`done` は同じ例外で reject するだけ。汲まない呼び手のために内部で 1 度
 * 握ってあるので、`done` を読まなくても unhandled rejection にはならない。
 */
export type GenerationStream = AsyncIterable<GenerationEvent> & {
  readonly done: Promise<GenerationStop>;
};

export type GenerationSequence = {
  /**
   * この会話が使える full スロットの容量（生成時に選んだ値 — 以後動かない）。
   *
   * 次のターンが通るかを自分で計算するときの上限がこれで、`program.capacity`（配布形の既定）
   * ではない（`capacity` を渡して作った sequence では 2 つが食い違う）。
   */
  readonly capacity: number;
  /**
   * この会話が既に占めている論理位置の数（= 次のターンの `prompt` が積み上がる起点）。
   *
   * MUST: **導出値**である（`context.pastLength` + 未 commit frontier 1 — ADR 0066 決定 6 の
   * 二重簿記の禁止）。独立した counter は持たないので、走行中に読むと**その時点で成功済みの
   * run まで**が反映される（生成が進むにつれ増える）。生成の合間に読めば「直近に完了した生成
   * までの確定値」で、切り詰めの判断はそこで行う。
   *
   * 次のターンが通るかは `used + prompt.length + maxNewTokens - 1 ≤ capacity` かつ
   * `used + prompt.length + maxNewTokens - 2 < program.maxPosition`（溢れたときの実値は
   * {@link GenerationCapacityError} が運ぶ）。
   */
  readonly used: number;
  /**
   * 1 ターンぶんを生成する。返り値を汲み切る（または `break` する）まで、この sequence の次の
   * `generate` / `dispose` は動き出さない（ADR 0083 決定 2 の直列化）。
   *
   * 受理集合の検査（`maxNewTokens` / prompt と `stopTokens` の token id / sampler の指定）と
   * **寿命**の検査（dispose 済みの sequence では生成できない）は**同期に**落ちる。予算の検査（位置表・容量）は
   * 自分の順番が来てからで、先行する生成が会話をどこまで進めるかが発行時点では決まっていない
   * ため（保存された counter から判断しない = 二重簿記の禁止）。
   *
   * MUST: 要求は**発行時に写す**（`prompt` / `stopTokens` の複製・`maxNewTokens` / `signal` の
   * 束縛・sampler 指定のスナップショット）。本体は最初の `next()` まで走らないので、写さないと
   * 「検査した値」と「実際に流す値」が別物になり得る（検査後に `prompt` へ語彙外 id を足す・
   * 走行中に `maxNewTokens` を伸ばす・`stopTokens` を空にする・`logitBias` を差し替える —
   * どれも例外にならない）。
   */
  generate(request: GenerationRequest): GenerationStream;
  /** context を返す（`generate` と同じ鎖に積むので、走行中の生成の後に走る）。 */
  dispose(): Promise<void>;
};

/**
 * 生成ループが context に要求する面。
 *
 * `greedy.ts` の `GenerationDisposable` を使わないのは `pastLength` が要るため — あちらは
 * 「論理長をホストが読まない」形（起点が prompt だけ）なので、意図的に寿命の返却しか持たない。
 */
export type GenerationContextFace = {
  /** 会話の論理長（この sequence の唯一の position の出どころ）。 */
  readonly pastLength: number;
  dispose(): Promise<void>;
};

/**
 * 生成ループが Session に要求する面（narrow interface — DI で fake を差せる）。
 *
 * MUST: `Pick<Session, …>` にはしない（`greedy.ts` の `GreedySession` と同じ理由 —
 * `GenerationContext` は `#` private を持つ名前的な型で、GPU 無しの fake が満たせない）。
 * context の型を型引数で通してあるのは「create が返した実体だけが run へ戻る」ことを型で
 * 縛るためで、`{ dispose }` に潰すと（メソッドの双変性で）別の context を実 Session へ渡す形が
 * 型検査を通ってしまう。
 *
 * MUST: 実 `Session` がこの面を満たすことはテスト側の型門で固定する（綴りのドリフト検出）。
 */
export type GenerationSession<C extends GenerationContextFace = GenerationContext> = {
  createGenerationContext(spec: GenerationContextSpec): Promise<C>;
  run(
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation: { readonly context: C; readonly queryLength: number },
  ): Promise<RunOutputs>;
};

export type GenerationSequenceOptions<C extends GenerationContextFace> = {
  readonly session: GenerationSession<C>;
  /** 検証済みの静的配線（`createGenerationProgram` の返り値）。 */
  readonly program: GenerationWiring;
  /**
   * この会話が使う full スロットの容量（省略時は {@link GenerationWiring.capacity} = 配布形の既定）。
   *
   * **sequence 生成時のノブ**である — 容量は state スロットの物理確保量そのもの（gemma4 E2B なら
   * full 層 12,288 B/token）なので、「長い会話」と「VRAM」の交換はここで 1 度だけ決まる。以後
   * 動かせないのは、context の物理確保が生成時に済んでいるためで、変えたいなら別の sequence を
   * 作る（KV は引き継げない — ADR 0066 追記 2 の `rewind` 全拒否と同じ線）。
   *
   * MUST: `program.chunkLength ≤ capacity ≤ program.maxPosition`（生成時に fail loudly）。
   */
  readonly capacity?: number;
};

/** i32 の入力テンソル 1 本（token id 列も絶対位置列も `[1, rows]`）。 */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** 行選択入力（`[1]` の i32 — ADR 0068 決定 4）。 */
const lastRowInput = (row: number): Tensor => ({
  dtype: "i32",
  shape: [1],
  data: Int32Array.of(row),
});

/**
 * 最終行 logits `[1,1,V]` の生データを読む。
 *
 * 名前と宣言形は program の setup が検証済みだが、ここでも見るのは「program が検証したのとは
 * **別のグラフ**で組まれた Session」を掴んだ場合の唯一の検出線だから（形が合う別の出力を掴むと
 * 例外も警告も出ないまま別の token 列が出る）。
 */
const readLogits = (
  outputs: RunOutputs,
  program: GenerationWiring,
  where: string,
): Float32Array<ArrayBuffer> => {
  if (!Object.hasOwn(outputs, program.logits)) {
    throw new Error(`${where}: グラフ出力 '${program.logits}' が無い`);
  }
  const tensor = outputs[program.logits];
  if (tensor.dtype !== "f32") {
    throw new Error(`${where}: '${program.logits}' が f32 でない（${tensor.dtype}）`);
  }
  const shape = tensor.shape;
  if (
    shape.length !== 3 || shape[0] !== 1 || shape[1] !== 1 || shape[2] !== program.vocabSize
  ) {
    throw new Error(
      `${where}: '${program.logits}' の形 [${
        shape.join(",")
      }] が [1,1,${program.vocabSize}] でない`,
    );
  }
  return tensor.data;
};

/**
 * このターンが踏む上限を run の**前**に見る（ADR 0083 決定 10）。
 *
 * - 位置は prefill が `past .. past+T-1`・decode が `past+T .. past+T+K-2` を踏む。
 * - `pastLength + queryLength` の最大は最後の decode の `past+T+K-1`（K=1 なら `past+T`）。
 */
const assertBudget = (
  program: GenerationWiring,
  capacity: number,
  pastLength: number,
  promptLength: number,
  maxNewTokens: number,
): void => {
  // 「今なら入る maxNewTokens」は 2 制約とも同じ形に畳める（{@link GenerationCapacityDetail}）。
  const detail = (
    constraint: GenerationCapacityConstraint,
    limit: number,
  ): GenerationCapacityDetail => ({
    constraint,
    pastLength,
    promptLength,
    requestedNewTokens: maxNewTokens,
    limit,
    maxNewTokens: limit - pastLength - promptLength + 1,
  });
  const lastPosition = pastLength + promptLength + maxNewTokens - 2;
  if (lastPosition >= program.maxPosition) {
    throw new GenerationCapacityError(
      `会話が位置表の外へ出る: 既存 ${pastLength} + prompt ${promptLength} + ` +
        `maxNewTokens ${maxNewTokens} は最終位置 ${lastPosition} を踏む` +
        `（この資産が引けるのは 0..${program.maxPosition - 1} — 会話の切り詰めはホストの責務）`,
      detail("maxPosition", program.maxPosition),
    );
  }
  const peak = pastLength + promptLength + maxNewTokens - 1;
  if (peak > capacity) {
    throw new GenerationCapacityError(
      `会話が state 容量を超える: 既存 ${pastLength} + prompt ${promptLength} + ` +
        `maxNewTokens ${maxNewTokens} は ${peak} 行を要求する` +
        `（容量 ${capacity} — 会話の切り詰めはホストの責務）`,
      detail("capacity", capacity),
    );
  }
};

/** 中断の例外か（`signal.reason` は包まずに throw するので、同一性で判別できる）。 */
const isAbortOf = (error: unknown, signal: AbortSignal | undefined): boolean =>
  signal !== undefined && signal.aborted && error === signal.reason;

/**
 * 1 会話ぶんの sequence を組む（context をここで確保し、以後の寿命はこの実体が持つ）。
 *
 * MUST: `GenerationContext` を外へ出さない（ADR 0083 決定 3）— 「最大 1 token の未 commit
 * frontier」を消費者から見せないことが決定 4 を成立させる要である。
 */
export const createGenerationSequence = async <C extends GenerationContextFace>(
  options: GenerationSequenceOptions<C>,
): Promise<GenerationSequence> => {
  const { session, program } = options;
  const capacity = options.capacity ?? program.capacity;
  if (!Number.isSafeInteger(capacity) || capacity < 1) {
    throw new Error(`capacity ${capacity} が 1 以上の整数でない`);
  }
  // 容量の関係は context を確保する前に見る（`parseGemma4PipelineConfig` が宣言に対して見るのと
  // 同じ 2 式を、実行時ノブに対しても通す）。
  if (capacity < program.chunkLength) {
    throw new Error(
      `capacity ${capacity} が chunkLength ${program.chunkLength} を下回る（1 chunk すら入らない）`,
    );
  }
  if (capacity > program.maxPosition) {
    throw new Error(
      `capacity ${capacity} が maxPosition ${program.maxPosition} を超えた` +
        `（容量いっぱいの会話がモデルの位置上限の外を引く）`,
    );
  }
  // MUST: 容量記号の束縛点は context 生成だけ（ADR 0066 追記 7 — run の bindings へは渡さない）。
  const context = await session.createGenerationContext({
    bindings: { [program.capacitySymbol]: capacity },
    chunkLength: program.chunkLength,
  });

  // 「generate 1 回ぶん」の直列化（ADR 0083 決定 2）— 自前ロックは作らない。
  const chain = createOperationChain();
  let pendingToken: number | undefined;
  let disposal: Promise<void> | undefined;

  /**
   * 直列化鎖の席を取り、返した関数で手放す。
   *
   * 席を取るのは**本体が回り出した時**（async generator の本体は最初の `next()` まで走らない）。
   * 発行時に取ると、汲まれないまま捨てられた iterable が鎖を永久に握り、`dispose` まで
   * 巻き添えになる。順序は「最初に汲み始めた方が先」。
   */
  const acquire = async (): Promise<() => void> => {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    await new Promise<void>((admitted) => {
      void chain(() => {
        admitted();
        return held;
      });
    });
    return release;
  };

  /**
   * ホスト由来の追加入力（宣言した名前と過不足なく一致することを毎回見る）。
   *
   * `signal` を降ろすのは、派生入力の材料が GB 級の遅延ロードになる配布形（gemma4 の PLE
   * sidecar）があるため — best-effort なので、無視する実装でも run の前の検査で閉じる。
   */
  const deriveInputs = async (
    ids: Int32Array<ArrayBuffer>,
    positions: Int32Array<ArrayBuffer>,
    signal: AbortSignal | undefined,
  ): Promise<RunInputs> => {
    const derived = program.derivedInputs;
    if (derived === undefined) return {};
    const extra = await derived.derive(
      [...ids],
      [...positions],
      signal === undefined ? {} : { signal },
    );
    const keys = Object.keys(extra);
    const missing = derived.names.filter((name) => !Object.hasOwn(extra, name));
    const surplus = keys.filter((name) => !derived.names.includes(name));
    if (missing.length > 0 || surplus.length > 0) {
      throw new Error(
        `derivedInputs が宣言と食い違う（欠け: ${missing.join(" / ") || "なし"} / ` +
          `余り: ${surplus.join(" / ") || "なし"}）`,
      );
    }
    return extra;
  };

  const generate = (request: GenerationRequest): GenerationStream => {
    // MUST: 寿命の検査も**同期**（ADR 0083 決定 3 — context を外へ出さないので、dispose 済みで
    // あることを呼び手が確かめる術がここ以外に無い）。遅らせると初反復まで落ちず、しかも
    // `GenerationContext` 側の汎用文言で出るため、真因（自分が dispose した）が読み取れない。
    if (disposal !== undefined) {
      throw new Error("GenerationSequence: dispose 済みでは生成できない");
    }
    // MUST: 要求はここで**写す**（{@link GenerationSequence.generate} の MUST）。async generator の
    // 本体は最初の `next()` まで走らないので、写さずに `request` を読み続けると受理集合の検査は
    // 「発行時の値」を、run は「汲み始めた時の値」を見る — 検査を通った要求が別物になって流れる。
    const prompt = [...request.prompt];
    const requestStopTokens = request.stopTokens === undefined ? [] : [...request.stopTokens];
    const maxNewTokens = request.maxNewTokens;
    const signal = request.signal;
    // 受理集合は同期に落とす（GPU にも順番待ちにも入る前）。写した後の値を見る。
    if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1) {
      throw new Error(`maxNewTokens ${maxNewTokens} が 1 以上の整数でない`);
    }
    // 停止 token は「出力に現れない id」なので、語彙外でも生成は**普通に完走してしまう**
    // （その id は抽選されないだけ）。効かない停止条件を静かに残さないため、program 側の
    // 集合（`createGenerationProgram`）と同じ値域門をここでも通す。
    const declared = new Set<number>();
    requestStopTokens.forEach((token, index) => {
      if (!Number.isSafeInteger(token) || token < 0 || token >= program.vocabSize) {
        throw new Error(`stopTokens[${index}] ${token} が語彙 0..${program.vocabSize - 1} の外`);
      }
      // 重複は「同じ条件を 2 度書いた」以上の意味を持てない = 呼び手の取り違えの徴候。
      if (declared.has(token)) throw new Error(`stopTokens に token ${token} が 2 度出る`);
      declared.add(token);
    });
    prompt.forEach((id, index) => {
      // `Int32Array` への書き込みは非整数の切り詰めも値域外の wrap も**黙って**行う
      // （2^32+1 → 1 = 別の有効 token id）ので、入口で落とす。語彙の外は embedding の
      // 範囲外 gather = 行ごと NaN 汚染になるので、同じ位置で見る。
      if (!Number.isSafeInteger(id) || id < 0 || id >= program.vocabSize) {
        throw new Error(`prompt[${index}] ${id} が語彙 0..${program.vocabSize - 1} の外`);
      }
    });
    // 抽選器は 1 生成に 1 つ（RNG 状態を step 越しに持つ）。指定の検査と、その指定の
    // スナップショット（`logitBias` の要素まで写す）も `createSampler` の中で済む。
    const sampler = createSampler(request.sampler);

    /**
     * 停止判定（配布形の集合と要求の集合の**和集合**）。
     *
     * 順序が意味を持つのは理由の側だけ — 両方に居る id は配布形の終端記号として `eos` で閉じる
     * （{@link GenerationStop} の doc）。
     */
    const stopFor = (token: number, tokens: number): GenerationStop | undefined => {
      if (isStopToken(token, program.stopTokens)) return { reason: "eos", token, tokens };
      if (isStopToken(token, requestStopTokens)) return { reason: "stop-token", token, tokens };
      return undefined;
    };

    let settle!: (stop: GenerationStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<GenerationStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // 二次的な通知路なので、読まれなくても unhandled rejection にしない（一次は iterable の throw）。
    done.catch(() => {});

    const events = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
      let stop: GenerationStop | undefined;
      let failure: { readonly error: unknown } | undefined;
      let release: (() => void) | undefined;
      /** 抽選した token の数（停止 token も 1 個 — `GenerationStop.tokens` の doc）。 */
      let generated = 0;
      try {
        release = await acquire();
        // 順番待ちの間に届いた中断は、ここで閉じる（先行の生成が長ければ待ちも長い）。同期の
        // 検査で足りるのは、待ち自体が `await` = 中断タスクの配送済みを意味するため。
        signal?.throwIfAborted();

        const past = context.pastLength;
        // 多ターンの連結（ADR 0083 決定 4）— 未 commit frontier を新 prompt の先頭へ。
        const promptIds = pendingToken === undefined ? prompt : [pendingToken, ...prompt];
        if (promptIds.length === 0) {
          throw new Error(
            "prompt が空（前ターンの pendingToken も無いので流す token が 1 つも無い）",
          );
        }
        assertBudget(program, capacity, past, promptIds.length, maxNewTokens);

        const chunks = planPrefillChunks(promptIds.length, program.chunkLength);
        // repetition penalty が見る「それまでの token 列」（HF が `input_ids` 全体に掛けるのと
        // 同じ形）。**このターンのぶんだけ**で、過去 turn は含まない（sequence の可変状態は
        // context と pendingToken の 2 つだけ = 会話全体の transcript は持たない）。
        const history = [...promptIds];

        let logits: Float32Array<ArrayBuffer> | undefined;
        for (const [index, chunk] of chunks.entries()) {
          await settleAbort(signal);
          // 有効行 1 本の chunk は decode 形（M=1）で流す — 計画を増やさず、かつ中断からの
          // 再開が「中断しなかった走り」と**同じ run** になる（`pendingToken` の再投入は
          // 常に 1 行なので、ここが多ターンのビット同一性の要）。
          const rows = chunk.queryLength === 1 ? 1 : program.chunkLength;
          const ids = new Int32Array(rows);
          const positions = new Int32Array(rows);
          // MUST: pad 行は 0 のまま（ADR 0066 追記 6 の値契約）。
          const base = context.pastLength;
          for (let row = 0; row < chunk.queryLength; row += 1) {
            ids[row] = promptIds[chunk.position + row];
            positions[row] = base + row;
          }
          const extra = await deriveInputs(ids, positions, signal);
          // MUST: 派生入力の `await` 明けにもう一度見る（ADR 0083 決定 5 の「段の境目」は run の
          // **発行直前**）。ここを省くと、中断が届いた後に run が 1 本まるごと進む — 先頭 chunk は
          // 常に cold miss で GB 級の shard を読むので、「送信直後に停止」で必ず踏む窓になる。
          signal?.throwIfAborted();
          const outputs = await session.run(
            {
              [program.inputIds]: i32Row(rows, ids),
              [program.lastRow]: lastRowInput(chunk.queryLength - 1),
              ...extra,
            },
            undefined,
            { context, queryLength: chunk.queryLength },
          );
          // 先頭 chunk が通った時点で frontier は KV に入った（= もう連結してはならない）。
          if (index === 0) pendingToken = undefined;
          logits = readLogits(outputs, program, `prefill@${chunk.position}`);
          yield { kind: "prefill", chunk: index + 1, chunks: chunks.length };
        }
        if (logits === undefined) throw new Error("prefill が 1 回も走っていない");

        // 生成の起点は最終 chunk の最終有効行（`last_row` で選んだ 1 行）。
        let token = sampler.next(logits, history);
        generated += 1;
        history.push(token);
        pendingToken = token;
        const stopped = stopFor(token, generated);
        if (stopped !== undefined) {
          stop = stopped;
          return;
        }
        yield { kind: "token", id: token, position: context.pastLength };

        // decode は「位置 P に `g_i` を置くと `g_{i+1}` が出る」形。回るのは `maxNewTokens - 1`
        // 回で、最後の token は未 commit のまま `pendingToken` に残る（決定 4）。
        for (let step = 0; step + 1 < maxNewTokens; step += 1) {
          await settleAbort(signal);
          const ids = Int32Array.of(token);
          const positions = Int32Array.of(context.pastLength);
          const extra = await deriveInputs(ids, positions, signal);
          // prefill 側と同じ理由で run の発行直前にもう一度見る（decode で踏むと token が 1 個
          // 余分に消費者へ届く）。
          signal?.throwIfAborted();
          const outputs = await session.run(
            {
              [program.inputIds]: i32Row(1, ids),
              [program.lastRow]: lastRowInput(0),
              ...extra,
            },
            undefined,
            { context, queryLength: 1 },
          );
          token = sampler.next(readLogits(outputs, program, `decode@${step}`), history);
          generated += 1;
          history.push(token);
          pendingToken = token;
          const stopped = stopFor(token, generated);
          if (stopped !== undefined) {
            stop = stopped;
            return;
          }
          yield { kind: "token", id: token, position: context.pastLength };
        }
        stop = { reason: "max-tokens", tokens: generated };
      } catch (error) {
        if (isAbortOf(error, signal)) stop = { reason: "aborted", tokens: generated };
        else failure = { error };
        // MUST: 包まずそのまま投げる（ADR 0083 決定 5 — 消費側が
        // `error === controller.signal.reason` で自分の中断を識別できる）。
        throw error;
      } finally {
        if (failure !== undefined) fail(failure.error);
        // `stop` が空のまま `finally` に来るのは `break` / `return()` 経由だけ。
        else settle(stop ?? { reason: "closed", tokens: generated });
        release?.();
      }
    };

    const iterable = events();
    return { [Symbol.asyncIterator]: () => iterable, done };
  };

  return {
    capacity,
    // MUST: getter で毎回導出する（`context.pastLength` と `pendingToken` が唯一の源）。
    get used(): number {
      return context.pastLength + (pendingToken === undefined ? 0 : 1);
    },
    generate,
    dispose(): Promise<void> {
      // MUST: 2 度目以降も同じ完了を返す（先に返すと呼び手が破棄前の窓を掴む）。
      disposal ??= chain(() => context.dispose());
      return disposal;
    },
  };
};
