/**
 * 1 会話ぶんの寿命を持つ生成の実体（ADR 0083 決定 1〜5 の `GenerationSequence`）。
 * **パイプライン非依存の共通処理**なので `program.ts` / `sampler.ts` と同じ `src/generation/` に置く。
 *
 * ## 可変状態は `context` と `pendingToken`（+ 投機の hidden の写しと未 commit 行数）
 *
 * MUST: sequence が持つ**進行の**可変状態は **`context` と `pendingToken` の 2 つだけ**である
 * （ADR 0083 決定 1 — 投機が足す hidden の写し・勘定・`frontierRows` は進行の記録ではない: 下の
 * 2 段落）。position / totalLength といった counter は持たず、run を組む直前に
 * `context.pastLength` を読む — 論理長の進行は run の成功で context が進める（ADR 0066 決定 6 の
 * **二重簿記の禁止**）。イベントに載せる `position` も、その run の**後**の `context.pastLength`
 * をその場で読んだ値で、保存しない。
 *
 * 投機経路（ADR 0096 段 3）はこれに **drafter へ渡す hidden の写し**（1 行 — readback は次の run
 * までしか有効でなく、導出できない）と勘定（`GenerationStop.speculation`）を足す。進行の記録は
 * 依然 `context.pastLength` だけで、verify は deferred run（論理長を保留）にし、**配送した token の
 * frontier まで**を `context.commit(rows)` で進める（下の「投機経路」節）。
 *
 * その `commit` に渡す行数（`frontierRows`）だけは sequence スコープの可変状態になる — 保留中の
 * run のどこまでを target が消費したかは context から導出できず（context が知るのは書いた行数
 * `queryLength` まで）、`used` の getter も配送中の論理長にこれを足す必要がある。進行の counter で
 * はない（commit の直後に 0 へ戻り、進行は `context.pastLength` にしか残らない）。
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
 *
 * ## 投機経路（ADR 0096 段 3 — `options.speculative` があるとき・温度に依らない）
 *
 * 1 cycle = draft（借り手 run・`k' = min(k, 残り予算 − 1)` 本）→ verify（貸し手の deferred run・
 * `[b, d₁..d_k']` の `k'+1` 行）→ 受理（同期・先頭一致・**停止 token で列挙を打ち切る** —
 * `speculation.ts`）→ 確定列を 1 個ずつ配送（`pendingToken` / `generated` / `history` は yield の
 * 前に更新 — 既存の MUST）→ **`settleCommit()` が frontier まで commit**（消費された行数 =
 * frontier にした token の数）。
 *
 * MUST: `settleCommit()` は配送ループの直後と generator の `finally` の両方で呼ぶ（保留があれば 1 回だけ
 * commit する）。消費者の `break` / `return()` は finally で「配送した token まで」を commit し、
 * verify の戻りから配送までの同期区間で例外が出れば `commit(0)`（frontier `b` は未投入のまま・
 * 棄却行は次の run が上書き）— どちらも「会話 = 消費者が受け取った列 + frontier 1 個」という
 * 非投機と同じ形に閉じる（保留を残すと context は dispose しか受け付けなくなる）。
 * MUST: 受理から最初の yield までに `await` を置かない（readback の subarray は次の run まで —
 * 貸し手 Session は他の sequence と共有されうる）。次 cycle の hidden は写す。
 * MUST: `k + 1 ≤ context.slidingSlack`（借り手は sliding ring の最古列まで読む — runtime の門は
 * `queryLength ≤ slidingSlack` で、`k` はその内側に取る）。
 *
 * ## 自己採算ゲート（ADR 0096 段 4-B ④ — `policy: "auto"` の既定）
 *
 * 投機の取り分は課題と host で決まり、負ける組み合わせがある（実測は `speculation-gate.ts` の
 * doc）。ゲートは cycle の壁と plain step の壁を自分で測り、負けている間は cycle の先頭で
 * `"plain"` を返す。その cycle は**既存の `drafted === 0` の経路**（verify 形 M=1・R=1・deferred・
 * `[b]` 1 行）をそのまま通す — 非投機の decode 経路へ分岐しないのは 3 つとも要るからである:
 * ① `readHidden` を通るので、次に speculate へ戻るときの drafter 入力（hidden）が繋がる
 * （decode 経路は logits しか読まない）② deferred なので、抽選が投げても `commit(0)` で frontier が
 * 二重投入されない ③ M=1・R=1 は PreparedPlan の鍵まで decode と同一で GPU 費用の差が無い。
 *
 * ゲート由来の plain step は**投機の勘定に入れない**（`cycles` / `draftRuns` / `drafted` /
 * `accepted` / `delivered` / `acceptedHistogram` の分母を汚さない）。数えるのは
 * {@link GenerationSpeculation.plainSteps} で、観測席には `{ kind: "decode", step }` として出る
 * （公開型に枝を足さない = 網羅 switch の消費者を壊さない）。予算末尾の強制 plain（`k' = 0`）は
 * 従来どおり verify として数え、ゲートには**諮らないし観測にも入れない**。
 *
 * MUST: 壁の観測に混ぜないのは 2 種類 — 各ターンの**最初の cycle**（PLE gather の cold miss を
 * 含む）と、**予算末尾の強制 plain**（呼び手の予算で形が決まった cycle で、投機の巧拙ではない）。
 * その 2 種類では観測の代わりに `SpeculationGate.skip()` を呼ぶ（探索の周期だけは進める — 混ぜ
 * ない cycle でも GPU の仕事は 1 本走っている）。
 */

import { closeableGenerator } from "../concurrency/closeable-generator.ts";
import { ModelInputError } from "../errors.ts";
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
import { disposeSteps } from "../session/dispose-steps.ts";
import { planPrefillChunks } from "./greedy.ts";
import { createSampler, isStopToken, type SamplerSpec } from "./sampler.ts";
import type { GenerationWiring } from "./program.ts";
import {
  acceptDrafts,
  createSpeculationTally,
  type DraftFace,
  planDraftLength,
  type SpeculationTally,
  takeDrafts,
  verifyRowIndices,
} from "./speculation.ts";
import {
  createSpeculationGate,
  type SpeculationDecision,
  type SpeculationGate,
  type SpeculationGateOptions,
} from "./speculation-gate.ts";

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
export class GenerationCapacityError extends ModelInputError {
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
    /**
     * この token が会話に置かれる絶対位置（非投機では = 直後の `context.pastLength`。投機では
     * 1 verify で複数 token が確定するので、commit 前の `pastLength + 1 + i`）。
     */
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
     * MUST: 停止 token（`eos` / `stop-token`）も 1 個として数える。非投機では抽選 1 回 = run 1 回
     * なので、この数がそのまま生成に費やした run 数と一致し、`tok/s` を再エンコード無しで書ける
     * （それがこの欄の目的）。投機では 1 verify run が最大 `k+1` 個を確定させるので run 数は
     * `speculation.cycles + speculation.draftRuns`（ゲート付きのターンは `+ plainSteps`）で読む。
     * 本文だけの数（= `token` イベントの数）が要るなら、停止 token を運ぶ 2 枝
     * （`"eos"` / `"stop-token"`）のとき 1 引く。
     *
     * `max-tokens` なら要求の `maxNewTokens` に一致し、`closed` / `aborted` では打ち切りまでに
     * 出した数になる（どちらも「成功した run のぶんだけ会話は進んでいる」— 上の節と同じ線。
     * 投機でも配送した token までしか commit しないので、この数 = 会話に入った生成 token 数）。
     */
    readonly tokens: number;
    /**
     * 投機の勘定（`options.speculative` を持つ sequence のターンだけ載る — 非投機の sequence では
     * 欄ごと無い）。
     */
    readonly speculation?: GenerationSpeculation;
  }
  & (
    | { readonly reason: "eos"; readonly token: number }
    | { readonly reason: "stop-token"; readonly token: number }
    | { readonly reason: "max-tokens" }
    | { readonly reason: "aborted" }
    | { readonly reason: "closed" }
  );

/**
 * 投機の勘定（{@link GenerationStop.speculation}）。
 *
 * 1 cycle あたりの確定数（token/cycle）の**正本は `delivered / cycles`** である。ふつうの cycle は
 * `1 + a` 個を確定させる（棄却でも frontier 1 個は必ず進む）が、受理した draft が停止 token だった
 * cycle はそこで列挙を打ち切るので `a` 個しか確定しない — 旧来の `(accepted + cycles) / cycles` は
 * その cycle 1 本につき分子が 1 だけ過大になる。予算末尾の `k' = 0` の cycle（draft を採らない）も
 * `cycles` に数え `acceptedHistogram[0]` に入る — draft あたりの受理数が要るなら `accepted / draftRuns`。
 */
export type GenerationSpeculation = {
  /** verify run の数（= cycle 数）。 */
  readonly cycles: number;
  /** draft run の数（`k' = 0` の cycle は draft を採らない）。 */
  readonly draftRuns: number;
  /** 出した draft の総数（Σ k'）。 */
  readonly drafted: number;
  /** 受理して配送した draft の総数（Σ a — 停止 token で打ち切った cycle は打ち切り後の数）。 */
  readonly accepted: number;
  /**
   * cycle が確定させた token の総数（Σ `confirmed.length` — token/cycle の分子）。
   *
   * 受理が決まった時点で cycle ぶんをまとめて積む（`accepted` と同じ位置）ので、`break` や中断で
   * 配送が cycle の途中で閉じたターンでは、消費者が受け取った数より多い。
   */
  readonly delivered: number;
  /** 添字 = その cycle の受理数 `a`（長さ `k+1`・`k' = 0` の cycle は添字 0）。 */
  readonly acceptedHistogram: readonly number[];
  /**
   * 自己採算ゲートが decode 形（M=1）で回した step の数（**このターンぶん**）。
   *
   * MUST: `policy: "always"`（ゲート無し）のターンでは**欄ごと無い** — `0` と「ゲートが居ない」は
   * 別物である（0 は「ゲートが 1 度も落ちなかった」という観測で、欄の不在は観測が無いこと）。
   * この step は上のどの欄にも入らないので、ターンの run 数は
   * `cycles + draftRuns + plainSteps` である。
   */
  readonly plainSteps?: number;
  /**
   * ゲートが speculate ↔ plain を切り替えた回数（**このターンぶん**・探索の 1 回試しは数えない）。
   *
   * ゲートそのものの寿命は sequence（移動平均をターンで捨てない）だが、この欄は他の欄と同じく
   * ターンの勘定である — ターン開始時の累計との差で出す。会話ぜんぶの累計が要るなら呼び手が
   * 足す（ターンぶんから累計は作れるが、累計からターンぶんは作れない）。
   */
  readonly switches?: number;
};

/**
 * 自己採算ゲートの状態（{@link GenerationRunPhase} の `gate` — `policy: "auto"` のときだけ載る）。
 *
 * MUST: 読むのはその run を**観測した後**である（観測でモードが倒れる cycle では、倒れた後の値が
 * 出る）— 「この run の壁を入れた結果どうなったか」が読み手の要る情報だからである。
 */
export type GenerationGateTrace = {
  /**
   * 定常モード（`SpeculationGate.mode`）。
   *
   * 探索で逆側を 1 回試している間も**定常側**を名乗る（探索バーストの cycle は
   * `mode: "plain"` の verify・`W1` プローブは `mode: "speculate"` の decode として出る）。
   */
  readonly mode: SpeculationDecision;
  /** このターン内の切替回数（{@link GenerationSpeculation.switches} と同じ数え方）。 */
  readonly switches: number;
  /**
   * ゲートがこの run を壁の観測に入れたか。
   *
   * `false` はゲートが `skip()` した cycle である（各ターンの最初の cycle と予算末尾の強制 plain —
   * モジュール doc の「混ぜない」2 種類）。壁は載っているが判定の材料にはなっていない。
   */
  readonly measured: boolean;
};

/**
 * run 1 本につき 1 通の観測（{@link GenerationSequenceOptions.onRun}）。
 *
 * 番号はすべて **1 始まり**。`prefill` の `chunk` / `chunks` は `GenerationEvent` の `prefill` と
 * 同じ数、`decode` の `step` はそのターンの decode run の番号、`draft` / `verify` の `cycle` は
 * 投機の cycle 番号（同じ cycle の draft と verify は同じ番号を名乗る）。`verify.rows` はその
 * run の有効行数 `k'+1`、`accepted` は受理した draft の数 `a`、`delivered` はその cycle が
 * 確定させた token 数（停止 token で列挙を打ち切った cycle だけ `accepted + 1` より少ない —
 * {@link GenerationSpeculation.delivered} と同じ量）。
 *
 * 投機ターンで自己採算ゲートが落とした plain step も `decode` を名乗る（`step` はそのターンの
 * plain step の通し番号 = {@link GenerationSpeculation.plainSteps} と同じ数え方）— run の形が
 * decode そのもの（M=1・R=1）だからで、枝を足すと公開型の網羅 switch を持つ消費者が壊れる。
 *
 * ## `wallMs` と `gate`（時間の内訳を読む 2 欄）
 *
 * `wallMs` は run 1 本ぶんの壁 ms で、投機の cycle は**先頭（draft の発行前）から受理判定の直後**
 * まで・非投機の decode は**step の先頭（派生入力の前）から抽選の直後**までを測る（区間は投機の
 * cycle と同じ — 派生入力を片方だけ含めると mode をまたいだ比較に系統差が乗る）。どちらも**配送の yield の
 * 前**に採るので、消費者の速さは入らない（ゲートが自分の判断に使う壁と同じ値である — 混ぜると
 * 遅い消費者ほど投機を切ることになる）。ゲートの有無に依らず常に載る。
 *
 * `gate` は `policy: "auto"` のターンにだけ載り、`kind` と `gate.mode` の組でその run の局面が
 * 決まる:
 *
 * | kind × gate.mode | 局面 |
 * | --- | --- |
 * | verify × speculate | 投機の定常（`switches` が 1 以上なら plain から戻った後） |
 * | verify × plain | plain 側の探索バースト |
 * | decode × speculate | speculate 中の `W1` プローブ |
 * | decode × plain | plain の定常（ゲートが落とした step） |
 */
export type GenerationRunPhase =
  | { readonly kind: "prefill"; readonly chunk: number; readonly chunks: number }
  | {
    readonly kind: "decode";
    readonly step: number;
    readonly wallMs?: number;
    readonly gate?: GenerationGateTrace;
  }
  | { readonly kind: "draft"; readonly cycle: number }
  | {
    readonly kind: "verify";
    readonly cycle: number;
    readonly rows: number;
    readonly accepted: number;
    readonly delivered?: number;
    readonly wallMs?: number;
    readonly gate?: GenerationGateTrace;
  };

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
 * 要求の**値域**検査（`maxNewTokens` / `stopTokens` / sampler の指定）— **検査の正本 1 本**。
 *
 * MUST: 同じ式を写して 2 本持たない。{@link GenerationSequence.generate} の同期区間と、
 * 高レベル面（`Gemma4Pipeline.chat` / `Gemma4ChatSession.send`）の**発行時**の両方がここを呼ぶ。
 * 高レベル面は `generate` を async generator の本体で呼ぶので、ここを自分で呼ばないと不正な
 * 要求が「最初の `next()` まで落ちない」（診断の位置が発行元から遠くなる）。
 *
 * NOTE: 見ないものが 2 つある — `prompt` の token id は**写した後**の配列に対して見る必要が
 * あり（写すのは `generate` の仕事）、予算（容量・位置上限）は自分の順番が来るまで確定しない
 * （`assertBudget`）。
 */
export const assertGenerationRequestValues = (
  vocabSize: number,
  request: Pick<GenerationRequest, "maxNewTokens" | "stopTokens" | "sampler">,
): void => {
  if (!Number.isSafeInteger(request.maxNewTokens) || request.maxNewTokens < 1) {
    throw new Error(`maxNewTokens ${request.maxNewTokens} が 1 以上の整数でない`);
  }
  // 停止 token は「出力に現れない id」なので、語彙外でも生成は**普通に完走してしまう**
  // （その id は抽選されないだけ）。効かない停止条件を静かに残さないため、program 側の
  // 集合（`createGenerationProgram`）と同じ値域門をここでも通す。
  const declared = new Set<number>();
  (request.stopTokens ?? []).forEach((token, index) => {
    if (!Number.isSafeInteger(token) || token < 0 || token >= vocabSize) {
      throw new Error(`stopTokens[${index}] ${token} が語彙 0..${vocabSize - 1} の外`);
    }
    // 重複は「同じ条件を 2 度書いた」以上の意味を持てない = 呼び手の取り違えの徴候。
    if (declared.has(token)) throw new Error(`stopTokens に token ${token} が 2 度出る`);
    declared.add(token);
  });
  // 抽選器の指定もここで落とす。`createSampler` は検査と写しと `Randu` の初期化だけで外部状態を
  // 触らないので、検査のために 1 度作って捨ててよい（実際に回す実体は `generate` が作る）。
  createSampler(request.sampler);
};

/**
 * token 列そのもの（`for await` で汲む）+ 停止理由。
 *
 * MUST: `done` は**二次的な**通知路である。失敗（run の失敗・容量超過）は iterable 側が throw
 * するのが一次で、`done` は同じ例外で reject するだけ。汲まない呼び手のために内部で 1 度
 * 握ってあるので、`done` を読まなくても unhandled rejection にはならない。
 * MUST: `done` は反復の終端（最後の `next()` が `done: true` を返す）より**前**に決着する —
 * 列を包む側（gemma4 の chat の締め）が、列が尽きた直後に `done` を待たずに読める
 * ことへ依存している。決着が終端より後ろへずれると、その `await` が消費側の `for await` ごと
 * 止まる。
 *
 * NOTE: 反復を**一度も始めずに** iterator の `return()` を呼ぶと本体が走らないので、`done` は
 * 決着しない（async generator の本体は最初の `next()` まで走らないため — 直列化の席も取らないので
 * 他の生成や `dispose` は妨げない）。`for await` では起きない形で、手で iterator を回す消費者
 * だけの注意である。
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
   * MUST: **導出値**である（`context.pastLength` + 未 commit の verify 行 + frontier 1 —
   * ADR 0066 決定 6 の二重簿記の禁止）。独立した counter は持たないので、走行中に読むと**その
   * 時点で会話に入った token まで**が反映される（生成が進むにつれ増える — `token` イベントの
   * 直後に読めば `used === position + 1`）。生成の合間に読めば「直近に完了した生成までの
   * 確定値」で、切り詰めの判断はそこで行う。
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
  /** deferred run の保留（無ければ `undefined`）— 投機経路の `settleCommit()` が読む。 */
  readonly pendingCommit: { readonly pastLength: number; readonly queryLength: number } | undefined;
  /** sliding ring の余裕（sliding スロットが無ければ `undefined`）— `k + 1 ≤ slidingSlack` の門。 */
  readonly slidingSlack: number | undefined;
  /** 保留中の deferred run の先頭 `rows` 行を確定させる（`0 ≤ rows ≤ queryLength`・同期）。 */
  commit(rows: number): void;
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
    generation: {
      readonly context: C;
      readonly queryLength: number;
      /** `"deferred"` = 論理長を進めず保留する（投機の verify — 受理数が決まってから `commit`）。 */
      readonly commit?: "deferred";
    },
  ): Promise<RunOutputs>;
};

/** 投機の指定（{@link GenerationSequenceOptions.speculative}）。 */
export type GenerationSpeculativeOptions<C extends GenerationContextFace> = {
  /**
   * 貸し手 context を受けて drafter の面を開く（sequence 生成時に 1 度・失敗したら貸し手 context
   * は sequence が畳んでから投げ直す）。畳む順は **drafter → 貸し手 context**（sequence が持つ）。
   */
  open(context: C): Promise<DraftFace>;
  /** 1 cycle で使う draft の本数（省略時は `DraftFace.steps`・`1 ≤ k ≤ steps`）。 */
  readonly k?: number;
  /**
   * 投機をいつ張るか（既定 `"auto"`）。
   *
   * - `"auto"` … 自己採算ゲート付き（モジュール doc の「自己採算ゲート」節）。負ける文脈では
   *   cycle ごとに decode 形（M=1）へ落ちる。
   * - `"always"` … 常に投機（ゲートを作らない = 壁時計を 1 度も読まない）。A/B の突合・検収の門・
   *   計測のための席で、`GenerationSpeculation` の `plainSteps` / `switches` も生えない。
   */
  readonly policy?: "auto" | "always";
  /** ゲートのノブ（`policy: "auto"` のときだけ効く — 省略時は `speculation-gate.ts` の既定）。 */
  readonly gate?: SpeculationGateOptions;
  /**
   * 壁時計（既定 `performance.now` — テストは偽時計を差す）。
   *
   * 読むのはゲートの観測と観測席の `wallMs`（{@link GenerationRunPhase}）で、`policy: "always"`
   * でも 1 cycle につき 2 回（cycle の先頭と受理判定の直後）呼ばれる。
   */
  readonly now?: () => number;
};

export type GreedyRunResult = { readonly value: number; readonly index: number };
export type GenerationGreedyRun<C extends GenerationContextFace> = (
  inputs: RunInputs,
  generation: { readonly context: C; readonly queryLength: number },
) => Promise<GreedyRunResult>;

export type GenerationSequenceOptions<C extends GenerationContextFace> = {
  /** 生logitsの最大値を返す内部能力。温度0・加工無し・非投機・観測無しのdecodeでのみ使う。 */
  readonly greedy?: GenerationGreedyRun<C>;
  readonly session: GenerationSession<C>;
  /** 検証済みの静的配線（`createGenerationProgram` の返り値）。 */
  readonly program: GenerationWiring;
  /** 投機的デコード（ADR 0096 段 3）。無ければ従来の 1 token = 1 run。 */
  readonly speculative?: GenerationSpeculativeOptions<C>;
  /**
   * run 1 本につき 1 回、その run の出力を読み終えた**同期区間**で呼ばれる観測席（verify は commit の
   * 直後）。診断（`Session.diagnostics()` の `lastRun*`）を「その run」の値として読めるのはこの
   * 同期区間だけである。無ければ何も呼ばない。
   */
  readonly onRun?: (phase: GenerationRunPhase) => void;
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

/**
 * この chunk を流す**物理行数**（宣言 shape の行数 = `M`）を選ぶ。
 *
 * - 有効行 1 本は **decode 形（M=1）**で流す。計画を増やさないうえ、中断からの再開が
 *   「中断しなかった走り」と**同じ run** になる（`pendingToken` の再投入は常に 1 行なので、
 *   ここが多ターンのビット同一性の要）。
 * - それ以外は `chunkBuckets` の中で `queryLength` 以上の**最小**の値。無ければ `chunkLength`。
 *
 * バケットを引く理由: 短い prompt を `chunkLength`（配布既定 768）行へ pad すると、pad 行は
 * 出力にも KV にも寄与しないのに行局所な op（linear / pointwise / norm）の仕事だけは物理行数に
 * 比例して積む。32 token の発話 1 本 + 1 token 生成の壁が 385 → 106 ms（linear −73%）になる実測
 * （`.claude/reviews/2026-09-06_performance-investigation/02_HOST_GENERATION.md` F-02）があり、
 * chat の user 発話は短いのに context の `chunkLength` は途中で変えられないので、多ターンでは
 * 毎ターンこの pad を払う。一方で長い prompt は 768 一括が最速なので「既定を下げる」形は採らず、
 * **許す物理行数を複数持って chunk ごとに選ぶ**（ADR 0066 決定 4 / 追記〈バケット〉）。
 *
 * MUST: 昇順の前提で先頭一致を採る（ここは順序を検査しない — 検査は
 * `createGenerationProgram` が通す runtime の `assertChunkBuckets` が持つ）。
 */
export const physicalChunkRows = (queryLength: number, program: GenerationWiring): number =>
  queryLength === 1
    ? 1
    : program.chunkBuckets.find((rows) => rows >= queryLength) ?? program.chunkLength;

/**
 * 行選択入力（`[R]` の i32 — ADR 0068 決定 4）。
 *
 * 渡すのは**行添字の列**である（要素数がその run の R を束縛する）。この経路が渡すのは常に
 * 1 本（prefill = 最終有効行 / decode = 行 0）なので、形は従来どおり `[1]` に畳まれる。
 */
const lastRowInput = (rows: readonly number[]): Tensor => ({
  dtype: "i32",
  shape: [rows.length],
  data: Int32Array.from(rows),
});

/** 選んだ行の logits の読み口（{@link readLogits} が返す — 行ごとの view）。 */
type LogitsRows = {
  /** 返ってきた行数（= `last_row` に渡した添字の本数 R）。 */
  readonly rows: number;
  /**
   * 行 `index` の生 logits（`vocabSize` 要素の view — 写さない）。
   *
   * MUST: 返るのは出力バッファの subarray なので、**次の run まで**しか有効でない
   * （readback バッファは run ごとに作り直される — 保存するなら呼び手が写す）。
   */
  row(index: number): Float32Array<ArrayBuffer>;
};

/**
 * 選んだ行の出力 `[1,R,width]`（logits なら `V`・hidden なら `H`）の生データを読む。
 *
 * 名前と宣言形は program の setup が検証済みだが、ここでも見るのは「program が検証したのとは
 * **別のグラフ**で組まれた Session」を掴んだ場合の唯一の検出線だから（形が合う別の出力を掴むと
 * 例外も警告も出ないまま別の token 列が出る）。
 *
 * MUST: 行数は**渡した添字の本数**（`expectedRows`）と突き合わせる。R はグラフでは記号なので、
 * 「1 行頼んだのに R 行返る」形も宣言としては正しく、ここが唯一の門である。
 */
const readRows = (
  outputs: RunOutputs,
  name: string,
  width: number,
  where: string,
  expectedRows: number,
): LogitsRows => {
  if (!Object.hasOwn(outputs, name)) {
    throw new Error(`${where}: グラフ出力 '${name}' が無い`);
  }
  const tensor = outputs[name];
  if (tensor.dtype !== "f32") {
    throw new Error(`${where}: '${name}' が f32 でない（${tensor.dtype}）`);
  }
  const shape = tensor.shape;
  if (
    shape.length !== 3 || shape[0] !== 1 || shape[1] !== expectedRows ||
    shape[2] !== width
  ) {
    throw new Error(
      `${where}: '${name}' の形 [${shape.join(",")}] が [1,${expectedRows},${width}] でない`,
    );
  }
  const data = tensor.data;
  return {
    rows: expectedRows,
    row: (index: number): Float32Array<ArrayBuffer> => {
      if (!Number.isSafeInteger(index) || index < 0 || index >= expectedRows) {
        throw new Error(`${where}: 行 ${index} が 0..${expectedRows - 1} の外`);
      }
      return data.subarray(index * width, (index + 1) * width);
    },
  };
};

/** 選んだ行の logits `[1,R,V]`。 */
const readLogits = (
  outputs: RunOutputs,
  program: GenerationWiring,
  where: string,
  expectedRows: number,
): LogitsRows => readRows(outputs, program.logits, program.vocabSize, where, expectedRows);

/** 選んだ行の最終 norm 後 hidden `[1,R,H]`（投機経路だけが読む — drafter の入力）。 */
const readHidden = (
  outputs: RunOutputs,
  program: GenerationWiring,
  where: string,
  expectedRows: number,
): LogitsRows => readRows(outputs, program.hidden, program.hiddenSize, where, expectedRows);

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
 * 投機の指定の受理集合（sequence 生成時・同期）。
 *
 * - `1 ≤ k ≤ steps`（配布形の drafter は `steps` 段で焼かれている）。
 * - `k + 1 ≤ slidingSlack`: verify は `k+1` 行を deferred で書き、棄却行が sliding ring の live 窓を
 *   潰さない条件は runtime の門 `queryLength ≤ slidingSlack`（借り手は最古列 `P−W` まで読む）。
 *   ここで見ないと runtime の門に落ちるのは最初の verify で、GB 級のロードの末になる。
 * - `k + 1 ≤ chunkLength` と「`k+1` 行以上のバケットが在る」: 無ければ `physicalChunkRows` が
 *   `chunkLength` 行（配布既定 768）へ落ち、例外なしで verify 1 本が 768 行になる。
 */
const assertSpeculativeSetup = (
  k: number,
  steps: number,
  slidingSlack: number | undefined,
  program: GenerationWiring,
): void => {
  // 段数は借り手の面が名乗る値（配布形の drafter が焼かれた段数）。非整数・0 以下だと `k` の
  // 検査が**素通りする**（`k` を省けば `k = steps` で NaN 比較が全部 false・`k` を渡せば
  // `k > steps` が false）ので、`k` より先に見る。
  if (!Number.isSafeInteger(steps) || steps < 1) {
    throw new Error(`drafter の段数 ${steps} が 1 以上の整数でない`);
  }
  if (!Number.isSafeInteger(k) || k < 1 || k > steps) {
    throw new Error(`speculative.k ${k} が 1..${steps}（drafter の段数）の外`);
  }
  if (slidingSlack !== undefined && k + 1 > slidingSlack) {
    throw new Error(
      `speculative.k ${k} は sliding ring の余裕 ${slidingSlack} に入らない（k + 1 ≤ 余裕 MUST）`,
    );
  }
  if (k + 1 > program.chunkLength) {
    throw new Error(
      `speculative.k ${k} の verify ${k + 1} 行が chunkLength ${program.chunkLength} を超える`,
    );
  }
  if (!program.chunkBuckets.some((rows) => rows >= k + 1)) {
    throw new Error(
      `verify ${k + 1} 行を載せるバケットが無い（chunkBuckets [${
        program.chunkBuckets.join(", ")
      }]）`,
    );
  }
};

/** 出力行の写し（readback の subarray は次の run まで — 次 cycle まで持つ hidden は写す）。 */
const copyRow = (row: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const copy = new Float32Array(new ArrayBuffer(row.byteLength));
  copy.set(row);
  return copy;
};

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
    // 許す物理行数は context 生成時にしか宣言できない（run 前検査が読む集合はここで畳まれる）。
    // 空配列 = 追加なしなので、宣言の無い配線でも従来どおりの 2 本になる。
    chunkBuckets: program.chunkBuckets,
  });

  /** drafter の面（投機の指定があるときだけ — 借り手は context の直後に開き、先に畳む）。 */
  let face: DraftFace | undefined;
  let k = 0;
  /**
   * 自己採算ゲート（`policy: "auto"` のときだけ）。
   *
   * MUST: 寿命は sequence である（ターンを跨いで持つ）— 壁と受理数の移動平均はターン 1 本では
   * 収束せず、会話ごとに測り直すと「最初の数 cycle だけ投機」を毎ターン繰り返す。
   */
  let gate: SpeculationGate | undefined;
  /** 壁時計（ゲートの観測と観測席の `wallMs` が読む — 既定は `performance.now`）。 */
  const now = options.speculative?.now ?? ((): number => performance.now());
  if (options.speculative !== undefined) {
    try {
      // ノブの門は借り手を開く**前**に通す（不正なノブが GB 級のロードの後まで落ちない）。
      if ((options.speculative.policy ?? "auto") === "auto") {
        gate = createSpeculationGate(options.speculative.gate);
      } else if (options.speculative.gate !== undefined) {
        // ゲートの居ない席にノブを渡すのは「効かないノブ」— 黙って通さない（横断不変条件）。
        throw new Error(
          `speculative: policy "always" にゲートのノブ ${
            JSON.stringify(options.speculative.gate)
          } を` +
            `渡したが、"always" にはゲートが居ない`,
        );
      }
      face = await options.speculative.open(context);
      k = options.speculative.k ?? face.steps;
      assertSpeculativeSetup(k, face.steps, context.slidingSlack, program);
    } catch (error) {
      // 借り手 → 貸し手の順で畳んでから投げ直す（開けなかった借り手は無い）。貸し手 context を
      // 漏らすと Session.dispose は気づかない（生きた context を数えていない）。後始末まで落ちたら
      // 両方を運ぶ（元の失敗だけにすると「context が返らなかった」ことが無音になる）。
      try {
        await disposeSteps([() => face?.dispose(), () => context.dispose()]);
      } catch (cleanup) {
        throw new AggregateError(
          [error, cleanup],
          "speculative.open が失敗し、貸し手 context の後始末も失敗した",
        );
      }
      throw error;
    }
  }

  // 「generate 1 回ぶん」の直列化（ADR 0083 決定 2）— 自前ロックは作らない。
  const chain = createOperationChain();
  let pendingToken: number | undefined;
  /**
   * 保留中の verify のうち frontier にした行数（= target が消費した行数・投機の配送中だけ非 0）。
   * {@link settleCommit} が保留中の verify をここまで commit する。
   *
   * MUST: sequence スコープに置く（`used` の getter が読む）。配送中の `context.pastLength` は
   * cycle の起点のままなので、これを足さないと**配送済みの token が論理長に現れない**。generate を
   * 跨いで残らないのは、`settleCommit()` が commit の直後に 0 へ戻し、配送の出口（停止 /
   * 配送ループの後 / `finally`）が全てそこを通るためである。`context.commit` 自身が投げると
   * 0 へ戻らないが、それは context が汚染 / dispose 済みのときだけで、以後は `pastLength` の
   * 読みも落ちる（残った値が観測されることはない）。
   */
  let frontierRows = 0;
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
    assertGenerationRequestValues(program.vocabSize, {
      maxNewTokens,
      stopTokens: requestStopTokens,
      sampler: request.sampler,
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
    const greedy = face === undefined && options.onRun === undefined && sampler.greedy !== undefined
      ? options.greedy
      : undefined;
    const pickGreedy = (result: GreedyRunResult): number => {
      if (
        !Number.isSafeInteger(result.index) || result.index < 0 || result.index >= program.vocabSize
      ) {
        throw new Error(`greedyの添字 ${result.index} が語彙 0..${program.vocabSize - 1} の外`);
      }
      if (sampler.greedy === undefined) throw new Error("greedy選択能力が無い");
      return sampler.greedy(result.value, result.index);
    };
    // 投機は drafter が居れば**温度に依らず**張る。受理は行ごとに `sampler.next` を非投機の decode と
    // 同じ logits・同じ history・同じ順で 1 回ずつ呼ぶ（確定 token 1 個につき 1 回）ので、RNG の消費列も
    // token 列も非投機と厳密に一致する — 温度 > 0 では「draft と同じ token を引いたら受理」が
    // one-hot draft の speculative sampling そのものになる（受理確率 = target 分布での draft の確率）。
    const tally: SpeculationTally | undefined = face === undefined
      ? undefined
      : createSpeculationTally(k);
    /** ゲートが decode 形で回した step の数（このターンぶん — 投機の勘定には入れない）。 */
    let plainSteps = 0;
    /**
     * ターン開始時のゲートの累計切替回数（差分が {@link GenerationSpeculation.switches}）。
     *
     * MUST: 鎖の席を取った**後**に読み直す（発行から本体が回り出すまでの間に、先行するターンが
     * ゲートを動かしうる）。発行時の値で初期化してあるのは、席を取る前に閉じたターン（順番待ちの
     * 中断）でも差が 0 になるようにするためである。
     */
    let switchesAtStart = gate?.switches ?? 0;
    const withSpeculation = (stopped: GenerationStop): GenerationStop =>
      tally === undefined ? stopped : {
        ...stopped,
        speculation: {
          ...tally,
          acceptedHistogram: [...tally.acceptedHistogram],
          // ゲートが居ないターンは欄ごと生やさない（0 と「ゲートが居ない」は別物）。
          ...(gate === undefined ? {} : { plainSteps, switches: gate.switches - switchesAtStart }),
        },
      };
    /** 保留中の verify を frontier まで確定させる（保留が無ければ何もしない）。 */
    const settleCommit = (): void => {
      if (context.pendingCommit === undefined) return;
      context.commit(frontierRows);
      frontierRows = 0;
    };
    const onRun = options.onRun;

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
        switchesAtStart = gate?.switches ?? 0;
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
        // 同じ形）。**このターンのぶんだけ**で、過去 turn は含まない（sequence は進行の counter も
        // 会話全体の transcript も持たない — モジュール doc の「可変状態」節）。
        const history = [...promptIds];

        let logits: LogitsRows | undefined;
        /** 最終 chunk の最終有効行の hidden（投機の最初の cycle の drafter 入力 — 写し）。 */
        let hidden: Float32Array<ArrayBuffer> | undefined;
        for (const [index, chunk] of chunks.entries()) {
          await settleAbort(signal);
          // 物理行数は有効行数から決める（decode 形 / バケット / chunkLength — 理由は
          // {@link physicalChunkRows}）。chunk の**割り方**は `chunkLength` のままで、
          // 変えるのは末尾 chunk を載せる行数だけである。
          const rows = physicalChunkRows(chunk.queryLength, program);
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
              // 選ぶのは最終有効行 1 本（R=1 — この経路は投機を張らない）。
              [program.lastRow]: lastRowInput([chunk.queryLength - 1]),
              ...extra,
            },
            undefined,
            { context, queryLength: chunk.queryLength },
          );
          // 先頭 chunk が通った時点で frontier は KV に入った（= もう連結してはならない）。
          if (index === 0) pendingToken = undefined;
          const where = `prefill@${chunk.position}`;
          logits = readLogits(outputs, program, where, 1);
          if (tally !== undefined) hidden = copyRow(readHidden(outputs, program, where, 1).row(0));
          onRun?.({ kind: "prefill", chunk: index + 1, chunks: chunks.length });
          yield { kind: "prefill", chunk: index + 1, chunks: chunks.length };
        }
        if (logits === undefined) throw new Error("prefill が 1 回も走っていない");

        // 生成の起点は最終 chunk の最終有効行（`last_row` で選んだ 1 行 = 行 0）。
        let token = sampler.next(logits.row(0), history);
        generated += 1;
        history.push(token);
        pendingToken = token;
        const stopped = stopFor(token, generated);
        if (stopped !== undefined) {
          stop = stopped;
          return;
        }
        yield { kind: "token", id: token, position: context.pastLength };

        if (tally !== undefined && face !== undefined) {
          // ---- 投機経路（モジュール doc の「投機経路」節・ADR 0096 段 3）
          if (hidden === undefined) throw new Error("prefill が hidden を出していない");
          const isStop = (id: number): boolean =>
            isStopToken(id, program.stopTokens) || isStopToken(id, requestStopTokens);
          /** 数える cycle の番号（ゲート由来の plain step は cycle 番号を消費しない）。 */
          let cycle = 0;
          /** このターンで回した cycle の本数（最初の 1 本を観測から外すための添字）。 */
          let turnCycles = 0;
          while (generated < maxNewTokens) {
            await settleAbort(signal);
            // k' = min(k, 残り − 1): 確定は最大 k'+1 個なので予算を超えない。k' = 0 は decode 1 行。
            const budget = planDraftLength(k, maxNewTokens - generated);
            // ゲートに諮るのは draft を採れる cycle だけ（予算末尾の強制 plain は投機の巧拙では
            // ないので、決定にも観測にも入れない）。MUST: `decide()` は 1 cycle に 1 回。
            const gated = budget >= 1 && gate?.decide() === "plain";
            const drafted = gated ? 0 : budget;
            if (gated) plainSteps += 1;
            else cycle += 1;
            /** この cycle を壁の観測に入れるか（モジュール doc の「混ぜない」2 種類）。 */
            const measured = gate !== undefined && turnCycles >= 1 && budget >= 1;
            turnCycles += 1;
            // 壁は cycle ごとに同じ 2 点（先頭 → 受理判定の直後）で採る。観測に入れるかで採る位置を
            // 変えないのは、偽時計のテストが実装の分岐をなぞるだけにならないようにするためである。
            // ゲートの有無にも依らない — 観測席の `wallMs` が同じ値を運ぶ（`GenerationRunPhase`）。
            const startedAt = now();
            let observerMs = 0;
            let drafts: number[] = [];
            if (drafted >= 1) {
              // drafter は (frontier b, b を出した行の hidden, b の位置 P) から d₁.. を出す。
              const raw = await face.draft({ token, hidden, position: context.pastLength });
              // 戻った run は名乗る（値域門で落ちる draft でも run は完了している）。
              tally.draftRuns += 1;
              tally.drafted += drafted;
              if (onRun !== undefined) {
                // 診断は直前の run に対応付けて即時配送する。その費用だけを採算から除く。
                const observerStartedAt = now();
                onRun({ kind: "draft", cycle });
                observerMs = now() - observerStartedAt;
              }
              drafts = takeDrafts(raw, drafted, program.vocabSize);
            }
            // verify = [b, d₁..d_k'] の k'+1 行を位置 P.. に置く（pad 行は 0 のまま）。物理行数 M は
            // k' が縮んでも `k+1` 行の形に固定する（R と同じく PreparedPlan の形を増やさない —
            // `k' = 0` だけは decode 形 M=1）。
            const queryLength = drafted + 1;
            const rows = physicalChunkRows(drafted === 0 ? 1 : k + 1, program);
            const ids = new Int32Array(rows);
            const positions = new Int32Array(rows);
            const base = context.pastLength;
            ids[0] = token;
            positions[0] = base;
            for (let index = 0; index < drafted; index += 1) {
              ids[index + 1] = drafts[index];
              positions[index + 1] = base + 1 + index;
            }
            const extra = await deriveInputs(ids, positions, signal);
            signal?.throwIfAborted();
            const rowIndices = verifyRowIndices(drafted, k);
            const where = gated ? `plain@${plainSteps}` : `verify@${cycle}`;
            const outputs = await session.run(
              {
                [program.inputIds]: i32Row(rows, ids),
                [program.lastRow]: lastRowInput(rowIndices),
                ...extra,
              },
              undefined,
              // MUST: deferred — 論理長は受理数が決まってから frontier まで進める（settleCommit）。
              { context, queryLength, commit: "deferred" },
            );
            // ---- ここから最初の yield までは同期（例外は finally の settleCommit が commit(0) で畳む）。
            frontierRows = 0;
            const logits = readLogits(outputs, program, where, rowIndices.length);
            const hiddenRows = readHidden(outputs, program, where, rowIndices.length);
            const { accepted, confirmed } = acceptDrafts(
              logits.row,
              drafts,
              sampler,
              history,
              isStop,
            );
            // ゲートの物差しは受理判定の直後、配送の前に採り、draft の診断費用を除く。
            // 消費者や観測の速さが混ざると、同じ推論でもゲートの判断が変わる。
            // MUST: 観測席へ載せる `wallMs` はこの同じ変数である（別に測り直すと、ゲートが見た壁と
            // 内訳の壁が食い違い、内訳から「なぜ倒れたか」を辿れなくなる）。
            const wall = now() - startedAt - observerMs;
            if (gate !== undefined) {
              // 混ぜない cycle は観測の代わりに `skip()`（ゲートの呼び出し規約 MUST）— 何も
              // 返さないと探索の周期が止まり、`W1` の初回サンプルがターンの 2 本目ではなく
              // 「観測に混ぜる最初の cycle」まで遅れる。
              if (!measured) gate.skip();
              else if (gated) gate.observePlain(wall);
              else gate.observeCycle(wall, confirmed.length);
            }
            // ゲートの状態は**観測の後**に読む（{@link GenerationGateTrace} の MUST）。
            const gateTrace: GenerationGateTrace | undefined = gate === undefined ? undefined : {
              mode: gate.mode,
              switches: gate.switches - switchesAtStart,
              measured,
            };
            // 次 cycle の drafter 入力 = 新しい frontier b' を出した行 a（写す — 次の run で消える）。
            // ゲート由来の plain step も verify 形の 1 行なので、ここで hidden が繋がる。
            const nextHidden = copyRow(hiddenRows.row(accepted));
            // 勘定に入れるのは投機の cycle だけ（ゲートの plain step は `plainSteps` が数える）。
            if (!gated) {
              tally.cycles += 1;
              tally.accepted += accepted;
              tally.acceptedHistogram[accepted] += 1;
              tally.delivered += confirmed.length;
            }
            // verify の観測は**この同期区間**（配送の yield をまたぐと、貸し手 Session を共有する
            // 別 sequence の run が挟まりうる）。commit は配送の後（frontier まで）。
            onRun?.(
              gated
                ? {
                  kind: "decode",
                  step: plainSteps,
                  wallMs: wall,
                  ...(gateTrace === undefined ? {} : { gate: gateTrace }),
                }
                : {
                  kind: "verify",
                  cycle,
                  rows: queryLength,
                  accepted,
                  delivered: confirmed.length,
                  wallMs: wall,
                  ...(gateTrace === undefined ? {} : { gate: gateTrace }),
                },
            );
            // `confirmed` は配送列そのもの（停止 token より後ろは受理の側で列挙していない）。
            for (let index = 0; index < confirmed.length; index += 1) {
              const id = confirmed[index];
              // MUST: frontier の更新は yield の前（`break` で finally へ入ったとき、frontier までが
              // commit される = 消費者の受け取った列 + frontier 1 個が会話）。
              token = id;
              generated += 1;
              history.push(id);
              pendingToken = id;
              frontierRows = index + 1;
              const stopped = stopFor(id, generated);
              if (stopped !== undefined) {
                stop = stopped;
                settleCommit();
                return;
              }
              yield { kind: "token", id, position: context.pastLength + 1 + index };
            }
            settleCommit();
            hidden = nextHidden;
          }
          stop = { reason: "max-tokens", tokens: generated };
          return;
        }

        // decode は「位置 P に `g_i` を置くと `g_{i+1}` が出る」形。回るのは `maxNewTokens - 1`
        // 回で、最後の token は未 commit のまま `pendingToken` に残る（決定 4）。
        for (let step = 0; step + 1 < maxNewTokens; step += 1) {
          await settleAbort(signal);
          // 壁は step の先頭（派生入力の前）から抽選の直後まで — 投機の cycle と同じ区間で、配送の
          // yield より前に採るので消費者の速さを含まない。観測席の `wallMs` だけが読む値である。
          const startedAt = now();
          const ids = Int32Array.of(token);
          const positions = Int32Array.of(context.pastLength);
          const extra = await deriveInputs(ids, positions, signal);
          // prefill 側と同じ理由で run の発行直前にもう一度見る（decode で踏むと token が 1 個
          // 余分に消費者へ届く）。
          signal?.throwIfAborted();
          const inputs: RunInputs = {
            [program.inputIds]: i32Row(1, ids),
            [program.lastRow]: lastRowInput([0]),
            ...extra,
          };
          // prefillは従来runを維持し、初回から大きなbackingを確保する費用を避ける。
          const result = greedy === undefined
            ? {
              kind: "logits" as const,
              outputs: await session.run(inputs, undefined, { context, queryLength: 1 }),
            }
            : {
              kind: "greedy" as const,
              selected: await greedy(inputs, { context, queryLength: 1 }),
            };
          // MUST: run が通った時点で frontier は KV に入った（= もう連結してはならない）— prefill
          // 側の `index === 0` と同じ理由で、抽選の**前**に落とす。ここを抽選の後に置くと、logits の
          // NaN などで `sampler.next` が投げたときに旧 frontier が残り、次ターンの prompt 先頭へ
          // 連結されて**同じ token が 2 つの位置に入る**（例外にならない沈黙劣化）。
          pendingToken = undefined;
          token = result.kind === "greedy" ? pickGreedy(result.selected) : sampler.next(
            readLogits(result.outputs, program, `decode@${step}`, 1).row(0),
            history,
          );
          onRun?.({ kind: "decode", step: step + 1, wallMs: now() - startedAt });
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
        try {
          // MUST: 保留中の verify を frontier まで commit する（`break` / `return()` / 例外のどれで
          // 来ても — 保留を残すと context は dispose しか受け付けない）。`commit` 自体は値域内の
          // 同期呼び出しで、ここで投げるなら簿記の破れ（黙って握らない — 鎖の席だけは返す）。
          settleCommit();
        } catch (cleanup) {
          // 元の失敗（device 消失で context が poison 済み、など）を上書きしない — 保留は
          // poison 済み context では読めず、dispose がまとめて捨てる。失敗が無いのに commit が
          // 落ちる形は簿記の破れで、finally からは投げられない（no-unsafe-finally）ので二次経路
          // （`done` の reject）で運ぶ。
          failure ??= { error: cleanup };
        } finally {
          if (failure !== undefined) fail(failure.error);
          // `stop` が空のまま `finally` に来るのは `break` / `return()` 経由だけ。
          else settle(withSpeculation(stop ?? { reason: "closed", tokens: generated }));
          release?.();
        }
      }
    };

    const iterable = closeableGenerator(events(), (failure) => {
      if (failure !== undefined) fail(failure.error);
      else settle(withSpeculation({ reason: "closed", tokens: 0 }));
    });
    return { [Symbol.asyncIterator]: () => iterable, done };
  };

  return {
    capacity,
    // MUST: getter で毎回導出する（`context.pastLength` と `frontierRows` と `pendingToken` が
    // 唯一の源）。投機の配送中は verify が保留のまま（`pastLength` は cycle の起点のまま）なので、
    // frontier までの未 commit 行を足す — これで**配送中でも
    // `used === 配送した token の position + 1`** が投機・非投機の別なく成り立つ（足さないと
    // cycle の途中では配送済みの token が抜け、次ターンの予算計算が最大 k だけ甘くなる）。
    get used(): number {
      return context.pastLength + frontierRows + (pendingToken === undefined ? 0 : 1);
    },
    generate,
    dispose(): Promise<void> {
      // MUST: 2 度目以降も同じ完了を返す（先に返すと呼び手が破棄前の窓を掴む）。
      // MUST: 借り手（drafter の面）→ 貸し手 context の順。失敗は集めて全段を回す（前段の
      // reject で後段が走らないと「二度と返せない context」が残る）。
      disposal ??= chain(() => disposeSteps([() => face?.dispose(), () => context.dispose()]));
      return disposal;
    },
  };
};
