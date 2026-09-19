/**
 * `Gemma4Pipeline` — **文字列 in → 文字列 out** の 1 本の面（生成 API 波の段 4）。
 *
 * 束ねるのは 4 つで、どれも既に別の場所で正本を持っている:
 *
 * 1. 製品グラフ（PLE 外出し + 最終行 logits 出口）の Session — `tools/export-recipes/gemma4/
 *    export_product.py` が書く shard 列
 * 2. ホスト PLE gather（`src/gemma/ple.ts` — ADR
 *    [0085](../../../../docs/decisions/0085-ple-host-gather.md)）を
 *    {@link GenerationWiring.derivedInputs} の席へ差す
 * 3. compile 済み tokenizer（`src/gemma/text/`）と chat フォーマット（`src/gemma/text/chat.ts`）
 * 4. 生成ループ（`src/generation/` — ADR
 *    [0083](../../../../docs/decisions/0083-generation-api-surface.md) の program / sequence /
 *    sampler）
 *
 * ここが足すのは**結線と id 空間の相互照合だけ**で、数値も語彙も 1 つも持たない。
 *
 * ## 2 つの入口（配布形と手元のバイト列）
 *
 * {@link Gemma4Pipeline.fromPretrained} は HF 配布形から（`karume.json` の `pipelineConfig` が
 * 静的配線を宣言する — 焼く側の正本は `tools/export-recipes/gemma4/distribution.py`）、
 * {@link Gemma4Pipeline.fromAssets} は取得済みバイト列から組む。**既定値は置かない** —
 * chunk 長も容量も位置上限も資産世代ごとに動くので、黙って古い数を使う形を作らない。
 *
 * MUST: PLE sidecar は配布形でも**全量常駐させない**（ADR 0085 決定 3）。`fromPretrained` は
 * shard を `assets` の遅延側で受け（`hub/components.ts` の `eagerAssets`）、触った 1 本だけを
 * 永続キャッシュから読み直す。
 *
 * ## MUST: id 空間を相互照合する（ADR 0085 決定 5）
 *
 * tokenizer が生成しうる id / 主 embedding の vocab 行数 / PLE sidecar の行数を
 * {@link admitGemma4} が突き合わせる。ここがずれると **OOB ではなく「別 token の有効な行」**を
 * 引く（例外なしで沈黙して壊れる）ので、fail loudly の門を置く場所はここしかない。
 *
 * ## MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）
 */

import { resolveGemmaSessionOptions } from "./session-options.ts";
import { createGemmaGreedyOutput, type GemmaGreedyOutput } from "./greedy-output.ts";
import { admitGemma4Qat, assertGemma4QatModel, assertGemma4QatPle } from "./qat.ts";
import { closeableGenerator } from "../concurrency/closeable-generator.ts";
import {
  acquireGpu,
  type AdmissionReport,
  assertChunkBuckets,
  estimateGraphMemory,
  type GenerationContext,
  type GpuContext,
  planWeightResidency,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type SharedWeight,
  type StateAttentionReduce,
} from "@karume/runtime";
import {
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  openAsset,
  type Quant,
  resolveFiles,
  type StreamAssetsOptions,
} from "@karume/hub";

import { createOperationChain } from "../concurrency/serial.ts";
import {
  assetComponentOpener,
  loadShardComponents,
  type ModelComponent,
  readCachedAsset,
} from "../hub/components.ts";
import { type FromPretrainedHubOptions, hubLoadOptions } from "../hub/load-options.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { assertRequiredLimitsBeforeDownload } from "../session/gpu-features.ts";
import {
  GEMMA4_PIPELINE_MAJOR,
  GEMMA4_PIPELINE_NAME,
  type Gemma4DefaultSampler,
  type Gemma4PipelineConfig,
  parseGemma4PipelineConfig,
} from "./config.ts";
import {
  createGenerationProgram,
  type GenerationGraph,
  type GenerationProgram,
  generationProgramFace,
  type GenerationWiring,
} from "../generation/program.ts";
import {
  assertGenerationRequestValues,
  createGenerationSequence,
  type GenerationEvent,
  type GenerationRequest,
  type GenerationRunPhase,
  type GenerationSequence,
  type GenerationSpeculation,
  type GenerationSpeculativeOptions,
  type GenerationStop,
  type GenerationStream,
  physicalChunkRows,
} from "../generation/sequence.ts";
import type { DraftFace } from "../generation/speculation.ts";
import {
  createSpeculationGate,
  type SpeculationGateOptions,
} from "../generation/speculation-gate.ts";
import { type SamplerSpec, snapshotSpec } from "../generation/sampler.ts";
import {
  createGemma4Ple,
  type Gemma4Ple,
  type Gemma4PleIndex,
  type Gemma4PleReadOptions,
  type Gemma4PleShardSource,
  parseGemma4PleIndex,
} from "./ple.ts";
import {
  GEMMA4_ROPE_LAYER_TYPES,
  GEMMA4_ROPE_PARTS,
  gemma4QatRopeInputs,
  gemma4RopeInputName,
  gemma4RopeInputNames,
  gemma4RopeInputs,
} from "./rope.ts";
import {
  admitGemma4Drafter,
  GEMMA4_DRAFT_STEPS,
  type Gemma4Drafter,
  type Gemma4DrafterAdmission,
  openGemma4DraftFace,
} from "./speculative.ts";
import {
  createStopStringFilter,
  type StopStringFilter,
  type StreamingDetokenizer,
} from "../text/detokenizer.ts";
import { parseGemmaTokenizerAsset } from "./text/asset.ts";
import { GemmaTokenizer } from "./text/tokenizer.ts";
import { type Gemma4ChatMessage, gemma4ChatPrompt, gemma4StopTokens } from "./text/chat.ts";

type GemmaFamily = "gemma4" | "gemma4-qat";

/**
 * グラフ入力の名前（正本は `export_product.py` の定数）。
 *
 * NOTE: `position_ids` はもう無い — RoPE の cos / sin 4 本（`./rope.ts` が名前も値も作る）が
 * 位置を運ぶ唯一の入力になった。位置は「表を引く添字」ではなく「表そのもの」として渡る。
 */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/**
 * 製品グラフの出口の本数（**順序が契約** — 出力 0 = 選んだ行の logits `[1,R,V]`・
 * 出力 1 = 同じ行の最終 norm 後 hidden `[1,R,H]`）。
 *
 * 名前ではなく順序で引く（`vocabSizeOf` / {@link buildGemma4Program}）— 出口の綴りは
 * 焼き手の内部名で、配布形ごとに動きうるためである。
 */
const GRAPH_OUTPUTS = 2;

/**
 * 配布形（manifest）の取得キー — weights 1 本と、全量で受け取る assets 2 本。
 *
 * MUST: PLE sidecar の shard は {@link EAGER_ASSETS} に**入れない**。1 本 250MiB 級 × 9 本で、全量常駐
 * させると ADR 0085 決定 3（触った shard だけ遅延ロード + LRU）そのものが成立しなくなる。
 * 取得キーは索引が書いたファイル名（`ple.json` の `shards[].file`）なので、遅延側の表は
 * 「eager に並べなかった残り」として自動的に PLE shard だけになる。
 */
const MODEL = "model";
/**
 * 投機（MTP drafter）を使うときだけ足す weights の役割（ADR 0096 段 2 §5）。
 *
 * MUST: 既定では**取得キーの表に載せない**（`resolveFiles` の `weights` で `model` だけに
 * 絞る）。載せると shard 面がこれを「用途不明の資産」として全量取得したうえ、遅延資産の
 * 突合（{@link assertPleShardAssets} は「遅延側 = PLE shard ちょうど」を要求する）が
 * **投機を使わないロードで**落ちる。
 */
const DRAFTER = "drafter";
const TOKENIZER_ASSET = "tokenizer";
const PLE_INDEX_ASSET = "ple_index";
const EAGER_ASSETS: readonly string[] = [TOKENIZER_ASSET, PLE_INDEX_ASSET];

/**
 * 取得済み資産から組むときの入力（**製品系列 1 世代ぶん**）。
 *
 * MUST: PLE sidecar だけ「バイト列」ではなく**読み口**を受ける。全量は i8 で 2,240MiB あり、
 * 常駐させると単一 ArrayBuffer 天井の議論（ADR 0085 決定 2）をホスト側で再現することになる。
 * 触った shard だけを遅延ロードする形（同 決定 3）が成立する唯一の受け方である。
 */
export type Gemma4Assets = {
  readonly config: Gemma4PipelineConfig;
  /** 製品グラフのコンテナ shard 列（**宣言順** — 先頭がグラフ shard。ADR 0081）。 */
  readonly model: readonly Uint8Array<ArrayBuffer>[];
  /** compile 済み tokenizer 資産のバイト列（ADR 0084 決定 1）。 */
  readonly tokenizer: Uint8Array<ArrayBuffer>;
  /** PLE sidecar の索引（`ple.json` のバイト列）。 */
  readonly pleIndex: Uint8Array<ArrayBuffer>;
  /**
   * PLE sidecar shard 1 本の**読み口を開く**（ファイル / hub の `openAsset` — 呼び手の責務）。
   *
   * 返す読み口は全量（`readAll`）が必須で、区間読み（`range`）は任意能力である。`range` を
   * 持たせない読み口では従来どおり「触った shard を全量読み → LRU 常駐」だけが起き、持たせると
   * decode の 1 token が 253MiB の全量読みではなく 9,100 B の 2 読みになる（ADR 0085 追記
   * 2026-09-07 — 方針表は `src/gemma/ple.ts` の `createGemma4Ple`）。
   *
   * 返した読み口の `readAll` / `range.read` が受ける `options.signal` は**その読みを起こした
   * 生成**の中断で、**best-effort**（無視しても壊れない — 中断が「この shard を読み終わって
   * から」効くだけ）。全量読みは 1 本 250MiB 級なので、対話的に止める使い方をするなら見る
   * 価値がある。
   *
   * open 自身が受ける `options.signal` も**その open を起こした生成**の中断で、開く動作が待つ
   * ぶん（hub の HF 取得元は在庫の無い参照で相 1 の温めを 1 度だけ挟む）に効く。
   *
   * MUST NOT: **開いた読み口が open 時の `options.signal` を保持しない**。handle は
   * pipeline の寿命ぶんキャッシュされるので、最初の生成の signal を握った読み口を作ると、
   * その生成が終わった後の読みが全部その中断に道連れになる。中断は読みごとの signal が担う。
   */
  readonly openPleShard: (
    file: string,
    options?: Gemma4PleReadOptions,
  ) => Promise<Gemma4PleShardSource>;
};

export type Gemma4PipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link Gemma4Pipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /**
   * PLE sidecar の常駐に使ってよい**ホスト RAM の上限（バイト）**（LRU — ADR 0085 決定 3）。
   *
   * 省略時は最大 shard 2 本ぶん（`ple.ts` の `defaultGemma4PleResidentBytes`）。
   * `0` は全量 shard も量子化行も保持しない。予算は値や token 列を変えない。
   *
   * NOTE: 本数ではなくバイトで受ける — shard 幅は資産世代で変わるので、「N 本」は世代ごとに
   * 違う RAM を意味する（ADR 0085 追記 2026-09-02）。区間読みが無い取得元は全量 shard の
   * LRU なので、予算を絞ると shard の読み直しが増える。
   * NOTE: **区間読みを持つ取得元**（`denoDirectory` など）では、空きに載る全量 shard と、
   * さらに残った空きの量子化行（最大 256 行）で予算を共有する。全量 shard を優先し、行の
   * 保存では shard を追い出さない。予算を絞ると行の再利用が減る場合がある（ADR 0085）。
   */
  readonly maxResidentPleBytes?: number;
  /**
   * 固定長 prefill chunk の行数（省略時は配布形の宣言 {@link Gemma4PipelineConfig.chunkLength}）。
   *
   * 上げるほど prefill の run 本数が減り（フェンス待ちの回数もその比で減る）、1 run あたりの
   * 一時バッファと attention のスコア行列が増える。グラフの chunk 行は記号なので、資産を
   * 焼き直さずに選べる。
   *
   * MUST: 2 以上・配布形の宣言 {@link Gemma4PipelineConfig.maxChunkLength} 以下・`maxPosition`
   * 以下（このパイプラインの門）。1 は decode 形の専用値で、prefill 形として流す経路が無い
   * （有効行 1 本の chunk は `GenerationSequence` が decode 形で流す）。上限を宣言から見るのは、
   * 記号 `M` の trace 範囲が資産からは読めない（IR の `symbols` は名前の列だけ）ため — 宣言が
   * 無かった頃は範囲外の値が例外なしで走っていた（2026-09-03 実測）。
   * 各 sequence の容量に対する `chunkLength ≤ capacity` は `createGenerationSequence` が見る。
   */
  readonly chunkLength?: number;
  /**
   * prefill 形として `chunkLength` に**加えて**使う物理 chunk 行数（省略時は
   * {@link GEMMA4_CHUNK_BUCKETS} のうち `chunkLength` 未満のもの・`[]` で無効）。
   *
   * 短い prompt を `chunkLength`（配布既定 768）行へ pad すると、pad 行は出力にも KV にも
   * 寄与しないのに行局所な op（linear / pointwise / norm）の仕事だけを物理行数に比例して積む。
   * バケットがあると、その chunk は `queryLength` 以上の最小バケット行だけで流れる。長い
   * prompt では 768 一括が最速なので、既定を下げるのではなく**実行形を増やす**形を採っている。
   *
   * MUST: 2 以上 `chunkLength` 未満の整数の**狭義昇順**（受理集合の正本は runtime の
   * `assertChunkBuckets` — この層は「どの入口の指定か」を文言に足すだけ）。
   *
   * NOTE: 本数ぶんだけ PreparedPlan の定常本数が増える（実行形 1 本 = 別鍵の計画 1 本 —
   * ADR 0042 決定 2 の LRU）。既定の 6 本 + prefill 形 + decode 形 = **1 つの容量あたり** 8 形で、
   * PreparedPlan の LRU 上限（runtime の `PREPARED_PLAN_CAPACITY` — verify 波で 12 へ広げた）に
   * 収まる — 鍵は解決済みスロット容量を含む（ADR 0066 決定 3）ので、容量の違う sequence を
   * 交互に回すと形は容量の数だけ倍になる。
   * さらに足すと、生成ループの中で最古が毎回落ちて decode が静かに再導出へ落ちる（例外は出ない —
   * 観測点は `SessionDiagnostics.lastRunPrepared`）。
   *
   * NOTE: 複数 chunk のターンでは末尾 chunk だけが別の M になるので、slot backing（容量 1）の
   * 作り直しが 1 回増える（prefill → decode の 2 回 / ターンが 3 回になる）。
   */
  readonly chunkBuckets?: readonly number[];
  /**
   * 実行 1 回ごとの診断を受け取る観測席（他 7 家族と同型）。op 別 GPU 時間（`lastRunTiming`）が
   * 要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す（ADR 0021 — 既定は計測しない）。
   *
   * 呼ばれるのは **run 1 本ごとに 1 通**（prefill は chunk ごと・decode は step ごと・投機では
   * cycle ごとに draft 1 通 + verify 1 通）で、その run の出力を読み終えた**同期区間**である
   * （診断の `lastRun*` が「その run」の値であるのはこの区間だけ）。何の run だったかは第 2 引数
   * {@link Gemma4RunPhase} が運ぶ — 回数から推定しない（複数 chunk の prefill があるターンでは、
   * 2 本目以降の prefill を decode と取り違える）。prefill 直後の最初の token は最終 chunk の
   * logits から抽選するだけで run を伴わないので、そこでは呼ばない。
   *
   * 診断を引く Session は phase で決まる — `draft` は借り手（drafter Session）・それ以外は
   * 貸し手（target Session）である。
   *
   * 中断せず走り切った非投機ターンの通知数は
   * `GenerationStop.tokens − 1 + prefill chunk 数`（`tokens` は停止 token も 1 個数えるので、
   * 引く 1 が「run を伴わない最初の抽選」ぶんである）。停止 token を引いた最後の decode run は
   * `token` を yield せずに終わるが、その run のぶんも 1 通届く。消費側の `break` / `return()` /
   * 中断で閉じたターンは、そこまでに完了した run のぶんだけが届く。投機ターンの通知数は
   * `prefill chunk 数 + speculation.cycles + speculation.draftRuns + speculation.plainSteps` で
   * ある（`plainSteps` = 自己採算ゲートが decode 形で回した step — `speculative: "always"` の
   * ターンでは欄ごと無い）。
   *
   * NOTE: 他ファミリと違ってコンポーネント名を渡さない — グラフが 1 本しかないので、名前が
   * 常に同じ 1 値になる（受け手が分岐できない引数を渡さない）。drafter が居るときも同じで、
   * どちらの Session の診断かは `phase.kind` から読む。
   *
   * コールバックの例外は握らない（fail loudly — そのターンごと落ちる）。
   */
  readonly onRunDiagnostics?: (
    diagnostics: SessionDiagnostics,
    phase: Gemma4RunPhase,
  ) => void;
  /**
   * states 形 attention ③PV の縮約形（省略時は {@link GEMMA4_STATE_ATTENTION_REDUCE} =
   * `"parallel"`）。
   *
   * `"parallel"` は KV 長方向を 16 レーンで分担する変種（perf-ledger K-12）で、decode の
   * attention が KV 長に比例して伸びる形を潰す（P=16K で 82 → 41 ms/token）。縮約順が違うので
   * `"sequential"`（runtime の参照経路）とビット同一ではないが、gemma4 の greedy / chat golden は
   * 両者で同一 — 既定への昇格はユーザーの品質裁定（2026-09-03）と golden の再走を同一コミットで
   * 行った（ADR 0058 決定 6・ADR 0067 追記 2026-09-03）。`"sequential"` は parity の突合や
   * 「順序依存の差を疑う」ときに戻す口。
   * `"parallel-fused"` は少数行・短い列上限で行統計と PV を融合する任意指定（ADR 0102）。
   * 加算順は parallel と同じで、対象外の形は parallel の経路を使う。
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /**
   * 量子化 GEMV の加算順。明示値 → 選択quantのsession → sequentialの順で解決する。
   * fromAssetsにはquantが無いため、未指定ならsequential。モデル構成による自動選択はしない。
   * 意味・適用形状はruntimeのSessionOptionsが正本。target / drafter両方へ渡す（ADR 0098）。
   * 特にQATでは生成列が変わる場合がある。
   */
  readonly linearGemvReduce?: SessionOptions["linearGemvReduce"];
  /** RMS→addの任意融合。意味・適用範囲はruntimeの同名設定が正本（ADR 0099）。 */
  readonly fuseRmsNormAdd?: SessionOptions["fuseRmsNormAdd"];
  readonly fuseLinearStaticQuantize?: SessionOptions["fuseLinearStaticQuantize"];
  /** RMSの任意縮約。subgroup32は対応GPU必須で、参照と加算順が変わる（ADR 0100）。 */
  readonly rmsNormReduce?: SessionOptions["rmsNormReduce"];
  /** GPUへの投入政策。target / drafterへ同じ値を渡し、省略時はruntimeの既定を使う。 */
  readonly submitPolicy?: SessionOptions["submitPolicy"];
  /**
   * slot backing（run の形ごとの中間バッファ束）を同時に保持するバイト予算。意味・既定・値域は
   * runtime の `SessionOptions.planBackingBudgetBytes` が正本で、この pipeline はそこへ素通しする
   * （`createSession` と {@link Gemma4Pipeline.estimateSessionMemory} の両方に同じ値が効く）。
   * 生成は prefill バケット形 ↔ decode 形の切替が毎ターン起きるので、既定（256 MiB）では
   * その両方が保持されて切替ごとの作り直しが消える。
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * 行ブロック gemv（linear の GEMV 族・M ≥ 2）の並列度目標。意味・既定（16384 = 参照 device の
   * 飽和点）・値域は runtime の `SessionOptions.linearGemvRowsThreadTarget` が正本で、この
   * pipeline はそこへ素通しする（target Session と drafter Session の両方に同じ値が効く）。
   * 飽和点が小さい GPU で下げると M=4 の verify が高い行ブロックを使う（重みの読み直しが減る）。
   */
  readonly linearGemvRowsThreadTarget?: number;
  /**
   * 投機デコード用の **MTP drafter を一緒に組む**（ADR 0096 — 省略時は組まない）。
   *
   * 指定すると配布形の `drafter` weights も取得し、target Session の埋め込み表 1 本を借りる
   * drafter Session を 1 本張る（バイトは複製されない）。**指定しない限り drafter の shard は
   * 1 バイトも落ちない** — 取得キーの表そのものから外れる（{@link DRAFTER} の MUST）。
   *
   * drafter が居る pipeline の `chat` / `sequence` は**既定で投機を張る**（1 verify run が
   * 最大 `k+1` token を確定させる）。ターン / 会話ごとに切るノブは
   * {@link Gemma4ChatOptions.speculative} と {@link Gemma4SequenceOptions.speculative}。
   * 投機は**温度に依らず**張り、出る token 列は非投機と同一である（受理は行ごとに、非投機の
   * decode が同じ位置で行う抽選と同じ logits・同じ history で 1 回ずつ引く — ADR 0096 決定 7）。
   *
   * MUST: `k` は **1..{@link GEMMA4_DRAFT_STEPS}**（配布形の drafter グラフが 3 段で焼かれていて、
   * 出口の本数が上限）。範囲外は fail loudly。
   */
  readonly speculative?: {
    /** 1 cycle で使う draft の本数（省略時は {@link GEMMA4_DRAFT_STEPS} = 配布形の段数）。 */
    readonly k?: number;
    /**
     * 自己採算ゲートのノブ（**計測・検収用の静的ノブ** — 既定で十分。
     * {@link Gemma4PipelineOptions.linearGemvRowsThreadTarget} と同じ扱いで、素通しする先は
     * `GenerationSpeculativeOptions.gate` である）。
     *
     * 効くのは**ゲートが居るターンだけ**である（`speculative: true` / 未指定 = `policy: "auto"`）。
     * 同じ pipeline の `speculative: "always"` のターンはゲートを作らないので、このノブは 1 つも
     * 降りない（`gate` 欄ごと生えない）— 1 本の pipeline で 3 モードを交互に回す A/B の台本
     * （`tools/mtp-bench`）がその形である。
     *
     * MUST: ノブの門は `fromPretrained` の引数検査の段で通す（{@link createSpeculationGate} を
     * 1 度作って捨てる）— 不正な値が GB 級の重みを読んだ後まで落ちない形にしない。
     */
    readonly gate?: SpeculationGateOptions;
  };
};

/**
 * gemma4 パイプラインが Session に与える ③PV の縮約形の既定
 * （{@link Gemma4PipelineOptions.stateAttentionReduce}）。
 *
 * MUST: runtime 側の既定（`"sequential"` = 参照経路 — ADR 0058 決定 2）は動かさない。既定を
 * 変えるのは「品質裁定を経た家族のパイプライン」だけで、低レベル面（`createSession` を自分で
 * 呼ぶ消費者・decode 系列の検収門）は参照経路のまま。
 */
export const GEMMA4_STATE_ATTENTION_REDUCE: StateAttentionReduce = "parallel";

/**
 * gemma4 パイプラインが使う prefill バケットの既定
 * （{@link Gemma4PipelineOptions.chunkBuckets}）— 2 冪の梯子。
 *
 * 実測で確定した列である（2026-09-07・RTX 3080 Ti / Vulkan・梯子 5 種 × prompt 5 長の ABBA）。
 * 効くのは短い発話で、20 token の prompt 1 本 + 1 token 生成の壁が `chunkLength` 768 固定の
 * 434 ms からこの梯子で 129 ms（−70%）になる。
 *
 * **512 を入れないのは実測で 768 より遅いからである**（310 行の prompt で 550 ms 対 411 ms）。
 * GEMM の幾何は `M ≤ 512` で `M64N32`・それより上で `reg128x128` へ切り替わる
 * （`gemm-geometry.ts`）ので、512 行の小タイル形は 768 行の大タイル形に負ける。256 も同じ
 * `M64N32` だが、130 行の prompt で 305 ms 対 410 ms と勝つので梯子に残す。
 *
 * **4 / 8 は投機の verify 用の段である**。verify は「draft した k 行 + 直前に確定した 1 行」を
 * 1 本の run で流す形（R = k+1 行）で、k=3 なら 4 行・k ≤ 7 なら 8 行に収まる（k の上限そのものは
 * sliding ring の余裕 = 棄却されうる行数の上限 8 で、context の `slidingSlack` が公開する）。
 * この段が無いと 4 行の verify が 32 行へ pad され、投機で削った仕事の一部をそのまま pad で
 * 払い直す。段そのものは通常の生成でも効く（4 token 以下の追い発話）。
 *
 * NOTE: 既定の 6 本 + prefill 形 + decode 形 = **1 つの容量あたり 8 形**が定常する
 * （{@link Gemma4PipelineOptions.chunkBuckets} の PreparedPlan の勘定）。
 *
 * MUST: 凍結する — この配列は module スコープの共有物で、消費者が並べ替えると以後に組む
 * pipeline の物理行数の選び方まで変わる（`chunkBuckets` は昇順前提で先頭一致を採る）。
 */
export const GEMMA4_CHUNK_BUCKETS: readonly number[] = Object.freeze([
  4,
  8,
  32,
  64,
  128,
  256,
]);

/**
 * 既定のバケット列を、選ばれた `chunkLength` に載る段だけへ切り詰める。
 *
 * `chunkLength` は実行時ノブ（{@link Gemma4PipelineOptions.chunkLength}）なので、既定の梯子を
 * そのまま渡すと `chunkLength: 64` のような指定が「既定同士の食い違い」で落ちる。切り詰めるのは
 * **既定だけ**で、呼び手が明示したバケットは 1 つも落とさず fail loudly させる（黙って捨てると
 * 「宣言したのに効かないバケット」が例外なしで残る）。
 */
const defaultChunkBuckets = (chunkLength: number): readonly number[] =>
  GEMMA4_CHUNK_BUCKETS.filter((rows) => rows < chunkLength);

/**
 * {@link Gemma4Pipeline.fromPretrained} が追加で受けるもの（選択軸 + 取得層へ透過するノブ）。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` が **HTTP 取得元専用**であることを含め、
 * 取得層のノブの説明は {@link FromPretrainedHubOptions} に 1 本化してある。
 */
export type Gemma4FromPretrainedOptions =
  & Gemma4PipelineOptions
  & FromPretrainedHubOptions
  & {
    /** manifest のモデル名（省略時は `defaultModel`）。 */
    readonly model?: string;
    /** quant 名（省略時はそのモデルの `defaultQuant`）。 */
    readonly quant?: string;
    /** 取得の中断（構築側へは渡らない — `chat` / `sequence` の中断は要求ごとの `signal`）。 */
    readonly signal?: AbortSignal;
  };

/** {@link Gemma4Pipeline.sequence} の指定（1 会話ぶんの寿命に効く唯一のノブ）。 */
export type Gemma4SequenceOptions = {
  /**
   * この会話が確保する full スロットの容量（省略時は配布形の既定 —
   * {@link Gemma4PipelineConfig.capacity}）。
   *
   * MUST: `chunkLength ≤ capacity ≤ maxPosition`（`createGenerationSequence` が fail loudly）。
   */
  readonly capacity?: number;
  /**
   * この会話で投機デコードを張るか（既定 = pipeline に drafter が居れば `true`）。
   *
   * - `true`（既定）… **自己採算ゲート付き**の投機。投機が decode に負けている間は cycle ごとに
   *   decode 形（M=1）へ落ち、勝てそうなら戻る（`src/generation/speculation-gate.ts`）。
   * - `"always"` … 常に投機（ゲートを作らない）。A/B の突合・検収の門・計測のための席である。
   * - `false` … 「drafter は載せたままこの会話だけ従来の 1 token = 1 run で回す」。
   *
   * drafter が居ない pipeline では `false` / 未指定でだけ通る（`true` / `"always"` は fail loudly —
   * 指定は黙って無視される値ではなく、**元から選択肢が無い**）。
   *
   * NOTE: ゲートは**壁時計**で切るので、既定席（`stateAttentionReduce: "parallel"`）では同じ
   * seed でも稀に出力が変わりうる（verify 形 M=4 と decode 形 M=1 で ①QK の縮約順が違い、近い
   * 値の token では argmax が割れる — `docs/limitations.md`）。厳密な再現性が要るなら
   * `"always"` か `stateAttentionReduce: "sequential"` を選ぶ。
   */
  readonly speculative?: boolean | "always";
};

/** {@link Gemma4Pipeline.estimateSessionMemory} の指定（見積る生成の形）。 */
export type Gemma4EstimateOptions = {
  /** 見積る容量（省略時はこの pipeline の既定）。 */
  readonly capacity?: number;
  /** 見積る chunk 長（省略時はこの pipeline が使う値）。 */
  readonly chunkLength?: number;
};

/**
 * 1 ターンぶんの chat リクエスト。
 *
 * MUST: 中身は {@link Gemma4Pipeline.chat} が**発行時に写す**（ADR 0083 追記 2026-09-02）— 返った
 * 列を汲み始めた後にこの object や `sampler` の指定を書き換えても、走行中のターンには効かない。
 */
export type Gemma4ChatOptions = {
  /** 生成する token 数の上限（1 以上）。停止 token はこの数に**含めない**。 */
  readonly maxNewTokens: number;
  /**
   * このターンだけ効かせる追加の停止 **token**（配布形の EOS 集合との和集合 —
   * `GenerationRequest.stopTokens`）。語彙外・重複は fail loudly。
   */
  readonly stopTokens?: readonly number[];
  /**
   * このターンだけ効かせる停止**文字列**（どれかが**復号後の本文**に現れた時点で止める）。
   *
   * 判定は復号の後（この層）で、token 境界を跨いで現れても止まる。一致した停止文字列そのものと
   * その後ろは**流れない**ので、`for await` で受けた片を連結したものが「停止文字列の手前まで」に
   * なる（`done` は `stop-string` とその綴りを運ぶ）。空文字列・重複は fail loudly。
   *
   * NOTE: 判定のために、停止文字列の**接頭辞になっている末尾**だけは確定していても出力を保留
   * する（`src/text/detokenizer.ts` の `createStopStringFilter`）。接頭辞でなくなった時点で
   * まとめて流れるので、止まらなかったターンの出力は 1 文字も欠けない。
   *
   * NOTE: 低レベル面（{@link Gemma4Pipeline.sequence}）にはこのノブが無い — `GenerationSequence`
   * は token id しか扱わないので、自分で回すなら停止は token で書く（`docs/limitations.md`）。
   */
  readonly stopStrings?: readonly string[];
  /** sampling の指定（省略時は {@link Gemma4PipelineConfig.sampler}、それも無ければ greedy）。 */
  readonly sampler?: SamplerSpec;
  /**
   * このターンの sequence が確保する容量（省略時は配布形の既定 —
   * {@link Gemma4PipelineConfig.capacity}）。
   *
   * `chat` は 1 ターン = 1 sequence なので、ここが「このターンの KV をどれだけ取るか」になる。
   * 長い会話を持ち回るなら `Gemma4ChatSession`（セッション 1 本ぶんの容量）を使う。
   */
  readonly capacity?: number;
  /**
   * prefill の進捗（chunk が 1 本 commit されるたび）。
   *
   * 長い prompt では最初の文字が出るまでの無音時間が prefill そのものなので、進捗を出す口が
   * ここにしか無い（`chat` が流すのは復号後の**本文**だけ — ADR 0084 決定 4）。
   *
   * コールバックの例外は握らない（fail loudly — そのターンごと落ちる）。
   */
  readonly onPrefill?: (progress: Gemma4PrefillProgress) => void;
  /**
   * 停止 token を除く生成 token の通知。復号・停止文字列の保留より前に同期で呼ぶ。
   * 本文を出さない特殊 token も含む。例外はそのターンへ伝播する。発行時に関数を写す。
   */
  readonly onToken?: (id: number) => void;
  /** 中断（段の境目で検査し `signal.reason` をそのまま throw する — ADR 0083 決定 5）。 */
  readonly signal?: AbortSignal;
  /**
   * このターンで投機デコードを張るか（既定 = pipeline に drafter が居れば `true` = ゲート付き —
   * 値の意味は {@link Gemma4SequenceOptions.speculative} と同じ）。
   *
   * 投機を張るかは sampler の指定に依らない（温度 > 0 でも token 列は非投機と同一）。この層は
   * 指定をそのまま降ろすだけで、cycle の組み立ては生成面が持つ。
   */
  readonly speculative?: boolean | "always";
};

/**
 * prefill の進捗 1 通ぶん（`chunk / chunks` がそのまま進捗）。
 *
 * `chunk` は**commit 済み**の chunk 数（1 始まり）で、`GenerationEvent` の `prefill` と同じ意味・
 * 同じ数である（この層は文字列の面なのでイベント型そのものを出さない）。
 */
export type Gemma4PrefillProgress = {
  readonly chunk: number;
  readonly chunks: number;
};

/**
 * 観測席（{@link Gemma4PipelineOptions.onRunDiagnostics}）が受ける「その 1 通がどの run か」。
 *
 * 席が受けるのは **run 1 本につき 1 通**で、この値はその run が何だったかを言う（順番の勘定
 * ではない）。番号はすべて 1 始まり — `prefill` の `chunk` / `chunks` は `GenerationEvent` の
 * `prefill` と同じ数（commit 済み chunk 数）、`decode` の `step` は**そのターンの** decode run の
 * 番号、`draft` / `verify` の `cycle` は投機の cycle 番号（同じ cycle の 2 本は同じ番号を名乗る）。
 *
 * MUST: 受け手は通知の回数ではなくこの値で分岐する — 複数 chunk に割れた prompt では
 * 「1 通目だけが prefill」が成り立たない。
 * MUST: 生成面の `GenerationRunPhase` **そのもの**である（写した型を持たない — 枝が片方だけ
 * 増えたときに型検査が通り続ける）。この層が足すのは「どちらの Session の診断を引くか」だけ。
 */
export type Gemma4RunPhase = GenerationRunPhase;

/**
 * chat 1 ターンの停止理由。
 *
 * sequence 層の理由（`eos` / `stop-token` / `max-tokens` / `aborted` / `closed`）に、この層でしか
 * 判定できない 1 つ（{@link Gemma4ChatOptions.stopStrings} の一致）を足したもの。`tokens` の
 * 意味は sequence 層と同じ（そのターンが生成した token 数 — 停止 token も 1 個）で、
 * `stop-string` では**停止文字列を含む片を出した token まで**が数に入る。
 */
export type Gemma4ChatStop =
  | GenerationStop
  | {
    readonly reason: "stop-string";
    /** 一致した停止文字列（出力には含まれない）。 */
    readonly stopString: string;
    readonly tokens: number;
    /**
     * 投機の勘定（`GenerationStop.speculation` をそのまま写したもの — 投機 sequence の
     * ターンだけ載る）。
     *
     * MUST: 写す。この枝は理由を差し替えるために object を組み直すので、写さないと
     * 「停止文字列で閉じたターンだけ勘定が消える」形になる（例外にならない欠落）。
     */
    readonly speculation?: GenerationSpeculation;
  };

/**
 * 文字列片の列（`for await` で汲む）+ 停止理由 + 一括で受け取る口。
 *
 * 片は逐次復号器が**確定させたぶん**だけで（ADR 0084 決定 4）、byte_fallback の途中は次の
 * token まで持ち越される。連結すると `decode(全 token id)` と一致する（停止文字列で切った
 * ターンだけは、その手前までになる）。
 *
 * MUST: **1 つのストリームは 1 通りにしか消費できない** — 反復（`for await`）と
 * {@link Gemma4ChatStream.text} の併用も、2 度の反復も、同期に throw する。生成は 1 度しか
 * 走らないので、2 通り目には「残り」しか流れない（先に汲んだ側だけが本文を持つ）— 例外に
 * ならない取り違えなので、口の側で塞ぐ。
 *
 * MUST: `done` は**二次的な**通知路である（`GenerationStream.done` と同じ規律）— 失敗は
 * iterable 側が throw するのが一次で、`done` は同じ例外で reject するだけ。
 */
export type Gemma4ChatStream = AsyncIterable<string> & {
  readonly done: Promise<Gemma4ChatStop>;
  /**
   * 汲み切って連結した 1 本の文字列（逐次表示が要らない呼び手の口）。
   *
   * 反復と同じ列を同じ順で汲むだけなので、`text()` の結果は「片を全部連結したもの」と一致する。
   * 停止理由が要るなら {@link Gemma4ChatStream.done} を併せて読む（`text()` の後でよい）。
   */
  text(): Promise<string>;
};

/** {@link Gemma4Pipeline} の内部状態（公開面には出さない）。 */
type Gemma4State = {
  readonly greedyOutput?: GemmaGreedyOutput;
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly session: Session;
  /** Session 構築に使った不変の設定。中間メモリ見積りにも同じ値を渡す。 */
  readonly stateAttentionReduce: StateAttentionReduce;
  /**
   * 製品グラフの宣言（`estimateSessionMemory` の材料 — ADR 0070 決定 5 の estimator は
   * `graph + 常駐計画` から純関数で出る）。
   *
   * MUST: `PreparedModel` ではなくグラフだけを持つ（`hub/components.ts` の同 MUST — 全量の
   * バイト列を掴んだままにしない）。
   */
  readonly graph: ModelComponent["graph"];
  /** 生成ループが読む内部配線（`createGenerationSequence` へ渡す実体）。 */
  readonly wiring: GenerationWiring;
  /**
   * 公開の読み口（{@link Gemma4Pipeline.program}）。
   *
   * 凍結した狭い面を getter のたびに作らないための席で、{@link buildGemma4Program} が返した
   * {@link Gemma4State.wiring} 1 本からその場で導いた値である（別経路で更新される欄ではない）。
   */
  readonly program: GenerationProgram;
  /**
   * PLE sidecar のホスト側キャッシュ（**{@link Gemma4Pipeline.dispose} の解放先**）。
   *
   * MUST: 席は dispose のためだけ — 引くのは `wiring.derivedInputs.derive` の閉包だけである。
   * どちらも {@link buildGemma4Program} の 1 回の返り値なので「片方だけ差し替えた」形は書けず、
   * 解放口を持たないと常駐ぶん（{@link Gemma4PipelineOptions.maxResidentPleBytes}）が
   * プロセス寿命まで残る。
   */
  readonly ple: Gemma4Ple;
  readonly tokenizer: GemmaTokenizer;
  readonly config: Gemma4PipelineConfig;
  /**
   * 投機の drafter（{@link Gemma4PipelineOptions.speculative} を渡したときだけ）。
   *
   * MUST: dispose は**貸し手（target Session）より先**（借り手が生きている間の貸し手
   * `dispose()` は runtime が fail loudly で断る — 借り手の bind group が貸し手の重みを
   * 掴んでいる）。
   */
  readonly drafter?: Gemma4Drafter;
  /**
   * 1 cycle で使う draft の本数（{@link Gemma4PipelineOptions.speculative} を渡したときだけ —
   * 省略時は {@link GEMMA4_DRAFT_STEPS}）。
   *
   * MUST: 席を持つのは投機の 2 つの消費者（生成の `speculative.k` と
   * {@link Gemma4Pipeline.estimateSessionMemory} の verify シナリオ `k+1` 行）が**同じ値**を
   * 見るため — 割れると「見積った形と違う run」が走る。
   */
  readonly speculativeK?: number;
  /**
   * 自己採算ゲートのノブ（{@link Gemma4PipelineOptions.speculative} の `gate` を渡したときだけ —
   * 省略時は `speculation-gate.ts` の既定）。
   *
   * MUST: 値の検査は pipeline を組む段（{@link assertSpeculative}）で済んでいる — この席は
   * 「生成のたびに素通しする」ためだけで、ここから先で解釈しない。
   */
  readonly speculativeGate?: SpeculationGateOptions;
  /**
   * Session に渡した slot backing の保持予算（{@link Gemma4PipelineOptions.planBackingBudgetBytes}・
   * 未指定なら runtime の既定）。
   *
   * MUST: 席を持つのは {@link Gemma4Pipeline.estimateSessionMemory} が**同じ値**を見積りへ渡す
   * ため — 握った値と見積りの前提が割れると、報告のピークが実際の保持集合と別の予算を名乗る。
   */
  readonly planBackingBudgetBytes?: number;
  /** 実行 1 回ごとの観測席（{@link Gemma4PipelineOptions.onRunDiagnostics}）。 */
  readonly onRunDiagnostics?: (
    diagnostics: SessionDiagnostics,
    phase: Gemma4RunPhase,
  ) => void;
};

/**
 * 家族 admission（GPU を取りに行く前・shard 面では重み prefetch の前に通す門）が確定させる材料。
 *
 * NOTE: PLE loader はここに載せない — `wiring.derivedInputs.derive` の閉包が持つのが唯一の
 * 参照で、席を 2 つ作ると「片方だけ差し替えた」形が書ける。
 */
type Gemma4Admission = {
  readonly component: ModelComponent;
  readonly config: Gemma4PipelineConfig;
  /** 最終行 logits 出口の語彙数（id 空間の相互照合の基準 — ADR 0085 決定 5）。 */
  readonly vocabSize: number;
  /** full スロットの容量記号（`createGenerationContext` の束縛点）。 */
  readonly capacitySymbol: string;
  /** 投機を指定したときだけ確定する drafter の材料（コンポーネント + 突合の結果）。 */
  readonly drafter?: {
    readonly component: ModelComponent;
    readonly admission: Gemma4DrafterAdmission;
  };
};

/**
 * 製品グラフ以外の資産（2 面が別の経路で用意し、解釈は 1 本に集める）。
 *
 * MUST: PLE sidecar だけ「バイト列」ではなく**読み口**を受ける（{@link Gemma4Assets} の同 MUST）。
 */
type Gemma4SidecarAssets = {
  readonly tokenizer: Uint8Array<ArrayBuffer>;
  /**
   * **解析済み**の PLE 索引。
   *
   * MUST: 解析は面ごとに 1 回だけ（`ple.json` を 2 度開かない）。`fromPretrained` は遅延資産
   * との突合にも索引が要るので、その 1 回をここへ持ち上げてある。
   */
  readonly pleIndex: Gemma4PleIndex;
  readonly openPleShard: (
    file: string,
    options?: Gemma4PleReadOptions,
  ) => Promise<Gemma4PleShardSource>;
};

/** `ple.json` のバイト列を索引へ落とす（fatal decode → JSON → 受理形）。 */
const parsePleIndexAsset = (bytes: Uint8Array<ArrayBuffer>): Gemma4PleIndex =>
  parseGemma4PleIndex(
    JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)),
  );

/**
 * PLE 索引が宣言する shard と、manifest の**遅延資産**が**ちょうど一致**することを見る門。
 *
 * MUST: 両方向を見る。索引にあって assets に無い shard は、その token 範囲を初めて引いた
 * ターン（= 会話の途中・3.7GiB のロード完了後）まで落ちない。assets にあって索引に無い
 * ファイルは「配布形が宣言した資産を 1 本も読まないまま動く」形で、永久に検出されない。
 * MUST: 呼ぶのは Session も重み shard も触る前（`#build` の前）。
 *
 * MUST: 「遅延資産 = PLE shard」が成り立つのは {@link EAGER_ASSETS} が tokenizer / ple_index の
 * 2 本ちょうどだからである（`deferred` は「eager に並べなかった残り」— `src/hub/components.ts`）。
 * eager を増やすときはこの式も直す。
 *
 * NOTE: `export` は門を直接叩くテストのため（実経路は配布形ミラー 3.7GiB のロードの後）。
 * `mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const assertPleShardAssets = (
  where: string,
  index: Gemma4PleIndex,
  deferredFiles: readonly string[],
): void => {
  const declared = new Set(index.shards.map((shard) => shard.file));
  const supplied = new Set(deferredFiles);
  const missing = [...declared].filter((file) => !supplied.has(file));
  const extra = [...supplied].filter((file) => !declared.has(file));
  if (missing.length === 0 && extra.length === 0) return;
  throw new Error(
    `${where}: PLE sidecar の索引と manifest の遅延資産が食い違う` +
      `（索引にあって assets に無い: ${missing.join(" / ") || "なし"} /` +
      ` assets にあって索引に無い: ${extra.join(" / ") || "なし"}）`,
  );
};

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする（7 家族と同じ門・同じ文言）。
 *
 * MUST: `slice` で写さない — 製品グラフの weight shard は 1 本 756MiB 級で、ホスト RAM の
 * ピークが倍になる。hub は buffer 全体を占める view を返す契約なので、崩れていたら
 * **取得層の不変条件破れ**として落とす。
 */
const assetBuffer = (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `${where}: 資産 '${key}' が無い（manifest の weights / assets に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `${where}: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

const assetBytes = (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): Uint8Array<ArrayBuffer> => {
  assetBuffer(where, assets, key);
  return assets[key];
};

/**
 * 選んだ行の logits 出口の語彙数をグラフから引く（`[1, R, V]` — ADR 0083 決定 6）。
 *
 * MUST: 呼び手に宣言させない。V は主 embedding の行数そのもので、宣言と食い違えば PLE
 * sidecar との相互照合（ADR 0085 決定 5）が**間違った基準**で通ってしまう。形の検査は
 * `createGenerationProgram` が同じ値でもう一度行う。
 *
 * MUST: 出口は**2 本ちょうど**（出力 0 = logits・出力 1 = 最終 norm 後 hidden）。順序は IR の
 * 契約で、名前で引かないのは配布形の綴りに依存しないためである（`capacitySymbolOf` と同じ
 * 流儀）。出口 1 本の旧配布形は**ここで**落とす — 互換分岐を書くと「hidden の無い資産で
 * 投機が黙って組めない」形が残る。
 */
const vocabSizeOf = (graph: GenerationGraph): number => {
  if (graph.outputs.length !== GRAPH_OUTPUTS) {
    throw new Error(
      `Gemma4Pipeline: グラフ出力が ${graph.outputs.length} 本` +
        `（製品グラフの出口は logits + hidden の ${GRAPH_OUTPUTS} 本 — ADR 0083 決定 6）`,
    );
  }
  const name = graph.outputs[0];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`Gemma4Pipeline: グラフ出力 '${name}' の値情報が無い`);
  }
  const shape = graph.values[name].shape;
  const vocab = shape[2];
  if (shape.length !== 3 || typeof vocab !== "number") {
    throw new Error(
      `Gemma4Pipeline: グラフ出力 '${name}' の shape [${shape.join(",")}] が [1,R,V] でない`,
    );
  }
  return vocab;
};

/**
 * 最終 norm 後 hidden 出口の幅をグラフから引く（`[1, R, H]` — 出力 1・ADR 0083 決定 6）。
 *
 * 呼ぶのは投機のときだけ（drafter の入力 `hidden` の幅がこれと一致する MUST）。本数と順序は
 * {@link vocabSizeOf} が既に見ている。
 */
const hiddenSizeOf = (graph: GenerationGraph): number => {
  const name = graph.outputs[1];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`Gemma4Pipeline: グラフ出力 '${name}' の値情報が無い`);
  }
  const shape = graph.values[name].shape;
  const hidden = shape[2];
  if (shape.length !== 3 || typeof hidden !== "number") {
    throw new Error(
      `Gemma4Pipeline: グラフ出力 '${name}' の shape [${shape.join(",")}] が [1,R,H] でない`,
    );
  }
  return hidden;
};

/**
 * full スロットの容量記号をグラフから引く。
 *
 * 記号は「入力 shape から決まらないもの」がちょうど 1 本のはずで（chunk 長の記号は
 * `input_ids` の 2 次元目から決まる・容量記号は states にしか現れない）、それを
 * `createGenerationContext` の束縛点へ渡す（ADR 0066 追記 7）。綴りを定数で持たないのは、
 * 資産側の綴りが変わったときに**黙って束縛されない記号**が残るのを避けるため。
 */
const capacitySymbolOf = (graph: GenerationGraph): string => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) {
      if (typeof dim === "string") fromInputs.add(dim);
    }
  }
  const free = graph.symbols.filter((symbol) => !fromInputs.has(symbol));
  if (free.length !== 1) {
    throw new Error(
      `Gemma4Pipeline: 入力 shape から決まらない記号が ${free.length} 本` +
        `（[${free.join(", ")}] — full スロットの容量記号 1 本であること）`,
    );
  }
  return free[0];
};

/**
 * RoPE 派生入力 4 本の宣言形（`[1, M, headDim]`）と `pipelineConfig.rope.<層種>.headDim` の突合。
 *
 * MUST: setup で見られる配線は setup で見る。`createGenerationProgram` が見るのは派生入力の
 * **名前の被覆**だけなので、幅の食い違い（層種別の取り違え = sliding 256 と full 512 の引き違い。
 * exporter 側 `rope.py` が `head_dim` / `global_head_dim` の分岐で自認している間違い方）は、
 * ホストが渡す表を初 `run` が受けるまで落ちない — 3.7GiB のロードの**後**で、しかも文言は
 * 「要素数が shape と合わない」になる。焼く側の鏡像は `export_decode.py` の `assert_rope_inputs`。
 *
 * NOTE: 内部の口だが export してあるのは、この単位なら宣言の突合を実 GPU も実資産も無しで
 * 縛れるため（`tests/gemma_config_test.ts` — siglip2 の `assertStaticDim` と同じ流儀）。
 */
export const assertRopeInputShapes = (
  graph: GenerationGraph,
  config: Gemma4PipelineConfig,
): void => {
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    const { headDim } = config.rope[layerType];
    for (const part of GEMMA4_ROPE_PARTS) {
      const name = gemma4RopeInputName(layerType, part);
      const input = graph.inputs.find((entry) => entry.name === name);
      if (input === undefined) {
        throw new Error(
          `Gemma4Pipeline: グラフ入力 '${name}' が無い（RoPE がホスト供給の資産でない）`,
        );
      }
      if (input.shape.length !== 3 || input.shape[2] !== headDim) {
        throw new Error(
          `Gemma4Pipeline: グラフ入力 '${name}' の shape [${input.shape.join(",")}] が` +
            ` pipelineConfig.rope.${layerType}.headDim ${headDim} と食い違う` +
            `（[1, M, ${headDim}] が要る）`,
        );
      }
    }
  }
};

/**
 * この製品グラフを gemma4 として実行できるかを見る（**重み shard を 1 バイトも取る前**）。
 *
 * MUST: 家族の門はこの 1 本に集める（他ファミリの `admit*` と同じ規律 — `hub/components.ts` の
 * {@link FamilyAdmission} 席で呼ばれる）。後段へ散らすと、shard 面では GB 級の重みを落とした
 * **後**にしか落ちない。
 *
 * NOTE: tokenizer / PLE sidecar の解析はここに置けない — admission の時点では assets を
 * まだ取っていない（取ってからでは重み prefetch より前という位置が保てない）ので、
 * {@link buildGemma4Program} に残る（anima の `#admit` と同じ分け方）。
 *
 * NOTE: `config` **単体**の検査はここには無い — 2 つの入口が**どちらも**
 * {@link parseGemma4PipelineConfig} を通してから呼ぶ（値域・関係・未知キーの門はそこが正本で、
 * 同じ検査を 2 実装持たない）。ここが見るのは宣言**とグラフの突合**だけで、
 * {@link assertRopeInputShapes} がその 1 本である（グラフはこの席で初めて手に入る）。
 */
const admitGemma4 = (
  component: ModelComponent,
  config: Gemma4PipelineConfig,
  drafter?: ModelComponent,
): Gemma4Admission => {
  const { graph } = component;
  assertRopeInputShapes(graph, config);
  const vocabSize = vocabSizeOf(graph);
  const capacitySymbol = capacitySymbolOf(graph);
  return {
    component,
    config,
    vocabSize,
    capacitySymbol,
    // drafter の門は `./speculative.ts` が持つ（借り物スロット・共有 initializer の綴りは
    // 投機の知識で、target の門とは別の 1 本）。target の材料は**確定したもの**を渡す。
    ...(drafter === undefined ? {} : {
      drafter: {
        component: drafter,
        admission: admitGemma4Drafter("Gemma4Pipeline", drafter.graph, {
          graph,
          rope: config.rope,
          hiddenSize: hiddenSizeOf(graph),
          capacitySymbol,
        }),
      },
    }),
  };
};

/**
 * {@link Gemma4PipelineOptions.speculative} の門（**資産を 1 バイトも取る前**に同期で落とす）—
 * 通れば解決済みの `k` を返す。
 *
 * MUST: `1 ≤ k ≤ {@link GEMMA4_DRAFT_STEPS}`。配布形の drafter グラフは 3 段で焼かれていて出口の
 * 本数が上限で、下は「draft を 1 本も採らない投機」= 意味を持たない指定である。範囲外を受けると
 * 「宣言と違う本数の draft を採る」形が黙って通る（生成面の `assertSpeculativeSetup` も同じ関係を
 * 見るが、そちらが落ちるのは GB 級のロードの**後**である）。
 *
 * ゲートのノブ（`gate`）の門も同じ段で通す — {@link createSpeculationGate} を 1 度作って捨てる。
 * 値域の判断はゲート自身が正本なので、同じ門を 2 実装持たない。
 */
const assertSpeculative = (
  where: string,
  speculative: NonNullable<Gemma4PipelineOptions["speculative"]>,
): number => {
  const k = speculative.k ?? GEMMA4_DRAFT_STEPS;
  if (!Number.isSafeInteger(k) || k < 1 || k > GEMMA4_DRAFT_STEPS) {
    throw new Error(
      `${where}: speculative.k ${k} が 1..${GEMMA4_DRAFT_STEPS} の外` +
        `（配布形の drafter は ${GEMMA4_DRAFT_STEPS} 段で焼かれている — 出口の本数が上限）`,
    );
  }
  if (speculative.gate !== undefined) createSpeculationGate(speculative.gate);
  return k;
};

/**
 * この会話で投機を張るなら、生成面へ渡す DI 一式を組む（{@link Gemma4SequenceOptions.speculative} /
 * {@link Gemma4ChatOptions.speculative} の解決 — chat と sequence が共有する 1 本）。
 *
 * 既定は「drafter が居れば張る」で、`false` だけが明示的な取り消しである。sampler の指定は
 * 見ない（投機は温度に依らず張り、token 列は非投機と同一 — ADR 0096 決定 7）。`"always"` は
 * 自己採算ゲートを作らない席（A/B・検収の門・計測）で、生成面の `policy` へそのまま降りる。
 *
 * pipeline のゲートのノブ（{@link Gemma4PipelineOptions.speculative} の `gate`）は
 * **`"auto"` のターンにだけ**降りる（`"always"` はゲートを作らない席なので、渡しても読む相手が
 * 居ない）。1 本の pipeline で `false` / `"always"` / `true` を交互に回すのが A/B の台本の形
 * （`tools/mtp-bench`）なので、この組み合わせ自体は誤りではない — ノブが効く範囲を
 * {@link Gemma4PipelineOptions.speculative} の doc と bench の README（`config.gate`）で名乗る。
 */
export const speculativeSetup = (
  state: Pick<Gemma4State, "drafter" | "speculativeK" | "speculativeGate">,
  enabled: boolean | "always" | undefined,
): GenerationSpeculativeOptions<GenerationContext> | undefined => {
  const drafter = state.drafter;
  if (enabled === false) return undefined;
  if (drafter === undefined) {
    // 未指定（undefined）は「drafter が居れば張る」、明示の指定は drafter を要求する — 黙って
    // 非投機で回すと、結果（`speculation` 欄の不在）からも無視を読み取れない。
    if (enabled !== undefined) {
      throw new Error(
        `speculative: ${JSON.stringify(enabled)} を渡したが、この pipeline は drafter 無しで` +
          `開かれている`,
      );
    }
    return undefined;
  }
  // ゲートのノブは `"always"` へ渡さない（ゲートが居ないので読む相手が無い）— 欄ごと生やさない
  // ことが「このターンにゲートは無い」の綴りである。
  const gate = enabled === "always" ? undefined : state.speculativeGate;
  return {
    // 借り手 context は sequence 生成時に 1 本開き、sequence の dispose が**貸し手より先**に畳む。
    open: (context: GenerationContext): Promise<DraftFace> => openGemma4DraftFace(drafter, context),
    k: state.speculativeK ?? GEMMA4_DRAFT_STEPS,
    policy: enabled === "always" ? "always" : "auto",
    ...(gate === undefined ? {} : { gate }),
  };
};

/**
 * 実行時ノブの `chunkLength` を検査して返す（{@link Gemma4PipelineOptions.chunkLength} の門）。
 *
 * MUST: 2 以上（グラフの chunk 記号は prefill 形の最小 2 で焼かれており、1 行の chunk は decode 形
 * として流れる）・配布形が宣言する `maxChunkLength` 以下・`maxPosition` 以下。宣言
 * （`parseGemma4PipelineConfig`）が同じ関係を既定値に対して見るので、ここが見るのは**呼び手が
 * 上書きした値**である。
 *
 * MUST: `maxChunkLength` の門は落とせない — 記号 `M` の trace 範囲は資産に残らない（IR の
 * `symbols` は名前の列だけ）ので、宣言だけが「この資産が受けられる chunk 行数」の出どころで
 * ある。門が無かった頃、上限 768 の資産に `chunkLength: 1024` を渡すと例外なしで走っていた
 * （2026-09-03 実測）— 保証の外で動く形は fail loudly にする（横断不変条件）。
 *
 * NOTE: 容量との関係（`chunkLength ≤ capacity`）はここでは見ない — 容量は sequence ごとに選ぶので、
 * 両者が揃う唯一の場所が `createGenerationSequence` である（同じ式を 2 箇所に持たない）。
 *
 * NOTE: `export` は門を直接叩くテストのため（{@link assertRopeInputShapes} と同じ扱い — 実経路は
 * `fromAssets` / `fromPretrained` / `estimateSessionMemory` の 3 つで、どれも妥当値しか渡さない）。
 * `mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const assertChunkLength = (
  chunkLength: number,
  config: Gemma4PipelineConfig,
): number => {
  if (!Number.isSafeInteger(chunkLength) || chunkLength < 2) {
    throw new Error(
      `Gemma4Pipeline: chunkLength ${chunkLength} が 2 以上の整数でない`,
    );
  }
  if (chunkLength > config.maxChunkLength) {
    throw new Error(
      `Gemma4Pipeline: chunkLength ${chunkLength} が配布形の宣言 maxChunkLength` +
        ` ${config.maxChunkLength} を超えた（記号 M を焼いた trace 範囲の外）`,
    );
  }
  if (chunkLength > config.maxPosition) {
    throw new Error(
      `Gemma4Pipeline: chunkLength ${chunkLength} が maxPosition ${config.maxPosition} を超えた`,
    );
  }
  return chunkLength;
};

/**
 * 実行時ノブの `chunkBuckets` を検査して返す（{@link Gemma4PipelineOptions.chunkBuckets} の門）。
 *
 * MUST: 受理集合の規則（2 以上 `chunkLength` 未満・狭義昇順）は**写さない** — 正本は runtime の
 * `assertChunkBuckets` 1 本で、そこが拒否する指定を context 生成まで通さないためにここで先に
 * 通す。この層が足すのは入口の名前だけで、`Gemma4PipelineOptions` に渡した呼び手が
 * 「自分のどの指定が落ちたか」を読めるようにする（`assertChunkLength` と同じ流儀）。
 *
 * NOTE: `maxChunkLength` の門は要らない（バケットは `chunkLength` 未満で、その `chunkLength`
 * 自体が {@link assertChunkLength} の門を通っている）。
 */
export const assertGemma4ChunkBuckets = (
  chunkBuckets: readonly number[],
  chunkLength: number,
): readonly number[] => {
  try {
    assertChunkBuckets(chunkBuckets, chunkLength);
  } catch (cause) {
    throw new Error(
      `Gemma4Pipeline: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
  }
  return chunkBuckets;
};

/**
 * admission を通った材料 + 資産から静的配線を組む（`fromAssets` と `fromPretrained` が共有）。
 *
 * ここが id 空間の相互照合（ADR 0085 決定 5）を全部通す — ①tokenizer が生成しうる id
 * ②主 embedding の vocab 行数 ③PLE sidecar の行数。
 */
const buildGemma4Program = (
  admitted: Gemma4Admission,
  assets: Gemma4SidecarAssets,
  options: Gemma4PipelineOptions,
  family: GemmaFamily,
): {
  readonly wiring: GenerationWiring;
  readonly tokenizer: GemmaTokenizer;
  readonly ple: Gemma4Ple;
} => {
  const { config, vocabSize, capacitySymbol, component } = admitted;
  if (family === "gemma4-qat") {
    assertGemma4QatPle(component.graph, assets.pleIndex);
  }
  const ropeInputs = family === "gemma4-qat" ? gemma4QatRopeInputs : gemma4RopeInputs;
  const tokenizer = new GemmaTokenizer(
    parseGemmaTokenizerAsset(assets.tokenizer),
  );
  // ① tokenizer が生成しうる id と ② 主 embedding の vocab 行数。
  if (tokenizer.maxTokenId >= vocabSize) {
    throw new Error(
      `Gemma4Pipeline: tokenizer の最大 token id ${tokenizer.maxTokenId} が` +
        ` 主 embedding の vocab 行数 ${vocabSize} の外（別の語彙で焼かれた組み合わせ）`,
    );
  }
  // ③ PLE sidecar の行数（この突合は `createGemma4Ple` が持つ — 同じ検査を 2 実装持たない）。
  const ple = createGemma4Ple({
    index: assets.pleIndex,
    openShard: assets.openPleShard,
    vocabSize,
    ...(options.maxResidentPleBytes === undefined
      ? {}
      : { maxResidentBytes: options.maxResidentPleBytes }),
  });

  const chunkLength = assertChunkLength(
    options.chunkLength ?? config.chunkLength,
    config,
  );
  const wiring = createGenerationProgram({
    graph: component.graph,
    inputIds: INPUT_IDS,
    lastRow: LAST_ROW,
    // 出口は順序で引く（{@link GRAPH_OUTPUTS} — 本数は admission が既に見ている）。
    logits: component.graph.outputs[0],
    hidden: component.graph.outputs[1],
    chunkLength,
    chunkBuckets: assertGemma4ChunkBuckets(
      options.chunkBuckets ?? defaultChunkBuckets(chunkLength),
      chunkLength,
    ),
    maxPosition: config.maxPosition,
    capacity: config.capacity,
    vocabSize,
    // 停止集合は tokenizer 資産の追加語彙から導出する（ADR 0083 決定 8 / 0084 決定 5 —
    // chat 形式と同じ digest set から来る）。
    stopTokens: gemma4StopTokens(tokenizer),
    capacitySymbol,
    // ホスト由来の per-chunk 入力の席に PLE gather と RoPE の cos / sin を差す（ADR 0085 / 本波）。
    // `options` は PLE へそのまま降ろす — shard 1 本 250MiB 級の読みが中断の届かない区間に
    // なるのを避ける（rope は同期の計算なので中断の窓を作らない）。
    derivedInputs: {
      names: [PER_LAYER_INPUTS, ...gemma4RopeInputNames()],
      derive: async (ids, positions, deriveOptions) => ({
        [PER_LAYER_INPUTS]: await ple.gather(ids, deriveOptions),
        ...ropeInputs(config.rope, positions),
      }),
    },
  });
  return { wiring, tokenizer, ple };
};

/**
 * この manifest を gemma4 として実行できるかを見る（**GPU も重み shard も触る前**）。
 *
 * MUST: 未知 major は fail loudly（ADR 0038 §1 — 「古い実装 × 新しいリポ」の沈黙劣化を止める
 * 唯一の門）。`quant` の実在検査は取得の前に済ませる（`resolveFiles` も同じことを見るが、
 * こちらは利用可能な一覧を添えて落とす）。
 *
 * MUST: 選ばれた `Quant` を**捨てずに返す** — `requiredLimits` の DL 前検査
 * （ADR 0089 決定 5）は呼び手（{@link Gemma4Pipeline.fromPretrained} の admission 閉包）が
 * 通す。ここで名前の実在だけ見て中身を落とすと、宣言された GPU 前提を誰も読まないまま
 * 3.7GiB を落とす形へ戻る。
 */
const gemma4ManifestConfig = (
  manifest: Manifest,
  selection: { readonly model?: string; readonly quant?: string },
  family: GemmaFamily,
): {
  readonly config: Gemma4PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
} => {
  const modelName = selection.model ?? manifest.defaultModel;
  if (family === "gemma4-qat") assertGemma4QatModel(modelName);
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `Gemma4Pipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== family) {
    throw new Error(
      `Gemma4Pipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${family}/${GEMMA4_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== GEMMA4_PIPELINE_MAJOR) {
    throw new Error(
      `Gemma4Pipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${family}/${GEMMA4_PIPELINE_MAJOR}）`,
    );
  }
  const quantName = selection.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `Gemma4Pipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  return {
    config: parseGemma4PipelineConfig(entry.pipelineConfig),
    quantName,
    quant: entry.quants[quantName],
  };
};

/**
 * 生成イベント → **確定した文字列片**（復号 → 停止文字列の判定）。返り値は一致した停止文字列
 * （`undefined` = 止まらずに列が終わった）。
 *
 * MUST: 停止文字列で止めるときは `return` で抜ける — `for await` の脱出はイベント列の
 * `return()` を呼ぶので、sequence は**中断（`break`）と同じ後始末**で畳まれる。畳み方を自前で
 * 書くと、KV の committed 整合（未 commit frontier 1 token）が 2 実装に分かれる。
 *
 * NOTE: 停止 token（sequence 層）と違い、停止文字列は復号の**後**でしか判定できない — 1 つの
 * 停止文字列が複数 token に割れることも、1 つの token が停止文字列の末尾と次の本文をまたぐ
 * こともあるため。だから席が 2 層に分かれる（ADR 0083 追記 2026-09-02）。
 *
 * NOTE: barrel（`mod.ts` / `./gemma`）には出さない**内部の口**である（公開の入口は
 * {@link Gemma4Pipeline.chat} だけ）。export してあるのは、この単位なら停止文字列の契約を
 * 実 GPU 無しで縛れるため（`tests/gemma_chat_test.ts`）。
 */
export const decodeChatChunks = async function* (
  events: AsyncIterable<GenerationEvent>,
  detokenizer: StreamingDetokenizer,
  stopStrings: StopStringFilter,
  onPrefill?: (progress: Gemma4PrefillProgress) => void,
  onToken?: (id: number) => void,
): AsyncGenerator<string, string | undefined, undefined> {
  for await (const event of events) {
    if (event.kind === "prefill") {
      // 文字列の面には prefill の片が無い（本文はまだ 1 文字も出ていない）ので、進捗だけを
      // 観測席へ渡す。例外は握らない（fail loudly — 呼び手のコールバックの誤りを飲まない）。
      onPrefill?.({ chunk: event.chunk, chunks: event.chunks });
      continue;
    }
    onToken?.(event.id);
    const chunk = stopStrings.push(detokenizer.push(event.id));
    if (chunk.text !== "") yield chunk.text;
    if (chunk.matched !== undefined) return chunk.matched;
  }
  // 復号器の持ち越し（byte_fallback の run）を確定させたぶんも判定へ通す — 停止文字列の最後の
  // 1 文字がその run の中に居ることがある。
  const tail = stopStrings.push(detokenizer.finish());
  if (tail.text !== "") yield tail.text;
  if (tail.matched !== undefined) return tail.matched;
  // 止まらずに終わったターンは、接頭辞として保留していたぶんを最後に流す（1 文字も落とさない —
  // 保留は判定のための遅延であって、出力の切り詰めではない）。
  const held = stopStrings.finish();
  if (held !== "") yield held;
  return undefined;
};

/**
 * 片の generator + 停止理由 → 公開の {@link Gemma4ChatStream}（**1 通りにしか消費できない**口）。
 *
 * MUST: 反復と {@link Gemma4ChatStream.text} は**同じ generator**を汲む（別経路を作らない）—
 * 生成は 1 度しか走らないので、一括の口が独自のループを持つと「どちらで読んだかで結果が違う」
 * 形が書けてしまう。2 通り目は静かに空を返すだけで例外にならないので、口の側で塞ぐ。
 *
 * NOTE: {@link decodeChatChunks} と同じく barrel には出さない内部の口である。
 */
export const chatStreamOf = (
  chunks: AsyncGenerator<string, void, undefined>,
  done: Promise<Gemma4ChatStop>,
): Gemma4ChatStream => {
  let claimed: "反復" | "text()" | undefined;
  const claim = (how: "反復" | "text()"): void => {
    if (claimed !== undefined) {
      throw new Error(
        `Gemma4ChatStream: 1 つのストリームは 1 通りにしか消費できない` +
          `（${claimed} で消費済み — ${how} は同じ生成をもう一度読もうとしている）`,
      );
    }
    claimed = how;
  };
  return {
    [Symbol.asyncIterator]: (): AsyncGenerator<string, void, undefined> => {
      claim("反復");
      return chunks;
    },
    done,
    // async にしない — 併用の検査は**同期に**落とす（返り値を await するまで気づけない形に
    // しない。`generate` の寿命検査と同じ規律）。
    text: (): Promise<string> => {
      claim("text()");
      return joinChunks(chunks);
    },
  };
};

/**
 * 観測席（{@link Gemma4PipelineOptions.onRunDiagnostics}）を生成面の `onRun` hook に仕立てる。
 *
 * 席が pipeline 層にあるのは、`GenerationSequence` が**パイプライン非依存**だからである
 * （Session も診断も知らない — ADR 0083）。生成面は run 1 本につき 1 回、その run の出力を
 * 読み終えた**同期区間**でこの hook を呼ぶので、この層がするのは「どちらの Session の診断を
 * 引くか」を `phase.kind` で決めることだけである。
 *
 * かつてはイベント列（`GenerationEvent`）を包んで run 数を**導出**していた（`withRunDiagnostics`）。
 * 導出は 2 つの例外を抱えていた — prefill 直後の最初の token は run を伴わない・停止 token を
 * 引いた最後の decode run は列に出ない（`done` から補っていた）— うえ、投機では 1 verify run が
 * 複数 token を出すので導出そのものが成り立たない。run の発行元が直接名乗る形（ADR 0083 追記
 * 〈hook〉）にすると、どちらの例外も消える。
 *
 * MUST: 観測席が無ければ `undefined` を返す（hook を渡さない = 生成面が 1 回も呼ばない）。
 * MUST: `draft` は**借り手**（drafter Session）の診断を引く。貸し手のものを渡すと、draft run の
 * 診断として「その前の verify run」の値が届く（例外にならない取り違え）。
 *
 * NOTE: 診断の型を型引数にしてあるのは、この関数が診断の**中身を 1 つも読まない**（席へ素通し
 * するだけ）ことを型で示すためで、同時に呼び出し規則の門（`gemma_chat_test.ts`）が実 Session
 * 無しで書ける。{@link Gemma4State} は `SessionDiagnostics` でそのまま満たす。
 * NOTE: `export` は門を直接叩くテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const runDiagnosticsHook = <D>(
  state: {
    readonly session: { diagnostics: () => D };
    readonly drafter?: { readonly session: { diagnostics: () => D } };
    readonly onRunDiagnostics?: (diagnostics: D, phase: Gemma4RunPhase) => void;
  },
): ((phase: Gemma4RunPhase) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  if (listener === undefined) return undefined;
  return (phase: Gemma4RunPhase): void => {
    if (phase.kind !== "draft") {
      listener(state.session.diagnostics(), phase);
      return;
    }
    const drafter = state.drafter;
    // draft run は drafter Session でしか起きない（居なければ簿記の破れ — 黙って貸し手の
    // 診断を渡すと、別の run の値が draft の名前で積算される）。
    if (drafter === undefined) {
      throw new Error(
        "Gemma4Pipeline: drafter が居ないのに draft run の観測が届いた",
      );
    }
    listener(drafter.session.diagnostics(), phase);
  };
};

/**
 * 停止文字列で閉じたターンの停止理由（`chat` と `Gemma4ChatSession.send` が共有する 1 本）。
 *
 * 理由と綴りはこの層の判定だが、`tokens` と `speculation` は**内側の値をそのまま写す**
 * （この層で数え直さない）。写す欄が増えたときに片方の入口だけ古いまま残るのを防ぐため、
 * 組み立てを 1 本にしてある。
 */
export const stopStringOf = (
  stopString: string,
  inner: GenerationStop,
): Gemma4ChatStop => ({
  reason: "stop-string",
  stopString,
  tokens: inner.tokens,
  ...(inner.speculation === undefined ? {} : { speculation: inner.speculation }),
});

/**
 * ターンの後始末 1 本（`chat` と `Gemma4ChatSession.send` が共有する）。
 *
 * MUST: `release` は**無条件に**呼ぶ。`cleanup`（sequence の返却・セッションの締め）が投げたら
 * 席を返さない形にすると、直列化鎖は前段の決着を得られないまま以後の `chat` / `dispose` を
 * 永久に待つ — 例外 1 つで二度と動かないパイプラインになる（device 消失時に `context.dispose`
 * が `flush` の失敗を伝播させる経路が実在する）。順序は flush-before-destroy のまま
 * 「`cleanup` → `release`」である。
 *
 * MUST: 本体（`failure`）も失敗しているときは**両方**運ぶ。呼び手の `finally` から呼ぶので、
 * ここで投げる例外は本体の例外を置き換える — 包まずに `AggregateError` へ 2 本とも載せる
 * （`errors[0]` が本体・`errors[1]` が後始末。中断の識別 `error === signal.reason` は
 * `errors[0]` に残る）。
 *
 * NOTE: 関数に切り出してあるのは、呼び手の `finally` に制御フロー文を置かないため
 * （`no-unsafe-finally` が禁ずるのは「元の例外を黙って捨てる」形で、ここは捨てずに畳んでいる）。
 */
export const closeChatTurn = async (
  where: string,
  failure: { readonly error: unknown } | undefined,
  cleanup: () => Promise<void>,
  release?: () => void,
): Promise<void> => {
  try {
    await cleanup();
  } catch (error) {
    if (failure === undefined) throw error;
    throw new AggregateError(
      [failure.error, error],
      `${where}: ターン本体と後始末の両方が失敗した`,
    );
  } finally {
    release?.();
  }
};

/** 後始末とリース返却が終わってから、iterable と同じ成否を done へ通知する。 */
export const completeChatTurn = async (options: {
  readonly where: string;
  readonly stream?: GenerationStream;
  readonly matched?: string;
  readonly failure?: { readonly error: unknown };
  readonly cleanup: (stop: Gemma4ChatStop | undefined) => Promise<void>;
  readonly release?: () => void;
  readonly settle: (stop: Gemma4ChatStop) => void;
  readonly fail: (error: unknown) => void;
}): Promise<void> => {
  let failure = options.failure;
  let stop: Gemma4ChatStop | undefined;
  try {
    const inner = options.stream === undefined
      ? { reason: "closed", tokens: 0 } satisfies Gemma4ChatStop
      : await options.stream.done;
    // 中断は iterable が reason を投げ、done は aborted を返す既存契約を維持する。
    if (failure === undefined || inner.reason === "aborted") {
      stop = options.matched === undefined ? inner : stopStringOf(options.matched, inner);
    }
  } catch (error) {
    failure ??= { error };
  }
  try {
    await closeChatTurn(
      options.where,
      failure,
      () => options.cleanup(stop),
      options.release,
    );
  } catch (error) {
    options.fail(error);
    throw error;
  }
  if (stop === undefined && failure !== undefined) {
    options.fail(failure.error);
    throw failure.error;
  }
  options.settle(stop ?? { reason: "closed", tokens: 0 });
};

/** 片を汲み切って連結する（{@link Gemma4ChatStream.text} の本体）。 */
const joinChunks = async (chunks: AsyncIterable<string>): Promise<string> => {
  let text = "";
  for await (const chunk of chunks) text += chunk;
  return text;
};

/**
 * gemma4 の chat パイプライン（製品グラフ 1 本 + ホスト PLE + tokenizer）。
 *
 * 構築の入口は {@link Gemma4Pipeline.fromPretrained}（HF から取得）と
 * {@link Gemma4Pipeline.fromAssets}（取得済みバイト列）の 2 つだけ — コンストラクタを private に
 * してあるのは、資産の突合を迂回した半端な状態を作れないようにするため（ADR 0008）。
 */
class GemmaPipeline {
  readonly #state: Gemma4State;
  /** chat と dispose の直列化鎖（1 つの Session を 2 本の会話で同時に押さない）。 */
  readonly #chain = createOperationChain();
  /** {@link Gemma4Pipeline.sequence} が渡した実体（dispose の取りこぼしを塞ぐ）。 */
  readonly #handed = new Set<GenerationSequence>();
  /** dispose の 1 本。**undefined でないことが「dispose 済み」**（派生状態を別に持たない）。 */
  #disposal: Promise<void> | undefined;

  protected constructor(state: Gemma4State) {
    this.#state = state;
  }

  /** 共通 factory の組み立て。family の契約を検査して状態を返す。 */
  protected static async loadPretrained(
    family: GemmaFamily,
    ref: string | HubRepoRef | DistributionSource,
    options: Gemma4FromPretrainedOptions = {},
  ): Promise<Gemma4State> {
    const where = family === "gemma4"
      ? "Gemma4Pipeline.fromPretrained"
      : "Gemma4QatPipeline.fromPretrained";
    if (family === "gemma4-qat" && options.speculative !== undefined) {
      throw new Error(`${where}: QAT の MTP は未対応`);
    }
    if (options.speculative !== undefined) {
      assertSpeculative(where, options.speculative);
    }
    const source = toManifestSource(
      ref,
      where,
      family === "gemma4"
        ? 'GEMMA4_SOURCES["gemma4"]（@karume/models/gemma）'
        : "明示した QAT 配布形の取得元",
    );
    const hubOptions: StreamAssetsOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const selection = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    // MUST: 取る weights を役割で絞る（`ResolveOptions.weights`）。投機を使わないロードで
    // drafter の shard が表に残ると、遅延資産の突合（{@link assertPleShardAssets}）が落ちる。
    const componentKeys = options.speculative === undefined ? [MODEL] : [MODEL, DRAFTER];
    const files = resolveFiles(loaded.manifest, {
      ...selection,
      weights: componentKeys,
    });
    const { admitted, assets, deferred } = await loadShardComponents(
      where,
      loaded,
      files,
      componentKeys,
      // 家族の門は admission 席で通す（重み shard を取る前 — `src/hub/components.ts`）。
      async (open) => {
        const { config, quantName, quant } = gemma4ManifestConfig(
          loaded.manifest,
          selection,
          family,
        );
        // 未対応の宣言を無視して走らせない。重みshardの取得より前に拒否する。
        const quantSession = resolveGemmaSessionOptions(
          quant.session,
          options,
          `${where}: quant '${quantName}'`,
        );
        const admitted = admitGemma4(
          open(MODEL),
          config,
          options.speculative === undefined ? undefined : open(DRAFTER),
        );
        if (family === "gemma4-qat") {
          admitGemma4Qat(
            admitted.component.graph,
            selection.model ?? loaded.manifest.defaultModel,
          );
        }
        // 配布形が宣言した `requiredLimits` は**重み shard を取る前**にここで見る
        // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
        // 他 7 家族と違って席が閉包側にあるのは、{@link admitGemma4} が構築オプションを
        // 受け取らない（グラフだけで決まる）ため。
        await assertRequiredLimitsBeforeDownload(
          quant.requiredLimits,
          options.gpu,
          `Gemma4Pipeline: quant '${quantName}'`,
        );
        return { ...admitted, quantSession };
      },
      {
        ...hubOptions,
        eagerAssets: EAGER_ASSETS,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      },
    );
    // MUST: 遅延側は PLE sidecar **ちょうど**であること。索引が知らないファイルが残っていれば
    // 「配布形が宣言した資産を 1 本も読まないまま動く」形で、逆に足りなければ会話の途中で
    // 初めて落ちる（どちらもロードの時点で分かる）。
    // 突合の本体は {@link assertPleShardAssets}（GPU も重み shard も触っていないこの位置で呼ぶ）。
    const pleIndex = parsePleIndexAsset(
      assetBytes(where, assets, PLE_INDEX_ASSET),
    );
    assertPleShardAssets(where, pleIndex, Object.keys(deferred));
    // MUST: 取得層のオプションから `signal` を落とす（`hub/components.ts` の相 2 と同じ理由 —
    // ロード 1 回の寿命を表す signal を、以後の生成が使う読み口へ持ち越さない）。載せ直すのは
    // **その読みを起こした生成**の signal だけで、寿命が読み 1 回と一致する。
    const { signal: _load, onProgress: _progress, ...streamOptions } = hubOptions;
    // open に渡す `signal` は**温めの中断**用で、返る handle はそれを保持しない（hub ⑧ の契約）。
    // HF 取得元は在庫の無い参照を開くとき相 1 の温めを 1 度だけ挟むので、全量 DL 1 本ぶんの待ちが
    // open に乗る — その待ちを起こした生成の signal で降りられるようにする。handle は pipeline の
    // 寿命ぶんキャッシュされる（`ple.ts` の `sources`）ので、handle 側が signal を保持していたら
    // **最初の**生成の中断で以後の読みが全部道連れになる。読み 1 回の中断は `readAll` /
    // `range.read` の `signal` が担う。
    const openPleShard = async (
      file: string,
      readOptions: Gemma4PleReadOptions = {},
    ): Promise<Gemma4PleShardSource> => {
      if (!Object.hasOwn(deferred, file)) {
        throw new Error(
          `${where}: PLE sidecar の shard '${file}' が manifest の assets に無い` +
            `（manifest が遅延資産として持つ shard: ${Object.keys(deferred).join(" / ")}）`,
        );
      }
      const ref = deferred[file];
      // 区間読みは**任意能力**（`openAsset` は取得元が持たなければ `undefined` を返す）。持たない
      // 取得元では `range` を生やさず、従来どおり全量読み + LRU へ倒れる。
      const reader = await openAsset(loaded, ref, {
        ...streamOptions,
        ...(readOptions.signal === undefined ? {} : { signal: readOptions.signal }),
      });
      return {
        // 行の位置検査は**宣言 size** で行う（実体長ではない — `Gemma4PleShardSource.bytes`）。
        bytes: ref.size,
        readAll: (options: Gemma4PleReadOptions = {}) =>
          readCachedAsset(where, loaded, ref, {
            ...streamOptions,
            ...(options.signal === undefined ? {} : { signal: options.signal }),
          }),
        ...(reader === undefined ? {} : {
          range: {
            cost: reader.cost,
            read: async (
              offset: number,
              length: number,
              options: Gemma4PleReadOptions = {},
            ) =>
              // hub が tight view を保証しているので、`buffer` がそのまま要求区間ちょうどになる。
              (await reader.read(offset, length, options)).buffer,
          },
        }),
      };
    };
    return await GemmaPipeline.#build(
      family,
      admitted,
      {
        tokenizer: assetBytes(where, assets, TOKENIZER_ASSET),
        pleIndex,
        openPleShard,
      },
      {
        ...options,
        ...admitted.quantSession,
      },
    );
  }

  /** 共通 factory の組み立て。family の契約を検査して状態を返す。 */
  protected static async loadAssets(
    family: GemmaFamily,
    input: Gemma4Assets,
    options: Gemma4PipelineOptions = {},
  ): Promise<Gemma4State> {
    const where = family === "gemma4"
      ? "Gemma4Pipeline.fromAssets"
      : "Gemma4QatPipeline.fromAssets";
    // MUST: この面に drafter の席は無い（{@link Gemma4Assets} が持つのは製品グラフ 1 本）。
    // 黙って投機なしで組むと「指定したのに効かない」形になるので fail loudly で断る。
    if (options.speculative !== undefined) {
      throw new Error(
        `${where}: speculative は受けられない` +
          `（Gemma4Assets に drafter の shard 列が無い — 投機は fromPretrained から組む）`,
      );
    }
    const config = parseGemma4PipelineConfig(input.config);
    if (input.model.length === 0) {
      throw new Error(
        `${where}: 製品グラフの shard 列が空（先頭がグラフ shard）`,
      );
    }
    const shards = Object.fromEntries(
      input.model.map((
        bytes,
        index,
      ): readonly [string, Uint8Array<ArrayBuffer>] => [
        `${MODEL}[${index}]`,
        bytes,
      ]),
    );
    const open = assetComponentOpener(
      where,
      shards,
      (key) => assetBuffer(where, shards, key),
    );
    // NOTE: `requiredLimits` の検査はこの面には無い — {@link Gemma4Assets} は manifest を
    // 持たない（バイト列と `config` だけ）ので、宣言そのものへ到達できない。実寸の検査は
    // Session 構築時の `assertWeightsWithinLimits`（ADR 0089 決定 1）が受け持つ。
    const admitted = admitGemma4(open(MODEL), config);
    if (family === "gemma4-qat") admitGemma4Qat(admitted.component.graph);
    return await GemmaPipeline.#build(family, admitted, {
      tokenizer: input.tokenizer,
      pleIndex: parsePleIndexAsset(input.pleIndex),
      openPleShard: input.openPleShard,
    }, options);
  }

  /**
   * admission を通った材料 + 資産から組む（2 面が共有する 1 本）。
   *
   * MUST: 資産の解析は **GPU を取りに行く前**（他 7 家族と同じ順序）— 順序がずれると、GPU の
   * 無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
   *
   * Session を 1 本持ち続けるのは siglip2 と同じ理由で、畳む相手（同時に載せられない別の
   * 巨大グラフ）が居ないため — 会話ごとに張り直すと 1.5GiB の重みを毎回アップロードし直す。
   *
   * 投機を指定したときは drafter Session も**ここで 1 本**張る（会話ごとではない — ADR 0096
   * 段 2 の裁定「束ね口は context」）。順序は target が先で、drafter はその埋め込み表を借りる。
   */
  static async #build(
    family: GemmaFamily,
    admitted: Gemma4Admission,
    assets: Gemma4SidecarAssets,
    options: Gemma4PipelineOptions,
  ): Promise<Gemma4State> {
    const { wiring, tokenizer, ple } = buildGemma4Program(
      admitted,
      assets,
      options,
      family,
    );
    // 投機の `k` は 2 つの消費者（生成と見積り）が同じ値を見るように**ここで 1 度**解決する。
    const speculativeK = options.speculative === undefined
      ? undefined
      : assertSpeculative("Gemma4Pipeline", options.speculative);
    const gpu = options.gpu ??
      await acquireGpu({
        subgroups: options.rmsNormReduce === "subgroup32" ||
          options.linearGemvReduce === "parallel-subgroup32",
      });
    const ownsGpu = options.gpu === undefined;
    // ③PV の縮約形は家族の既定（K-12 昇格済み）— 呼び手が明示すればそれに従う。予算は
    // 未指定なら欄ごと渡さない（既定値をここに写すと、runtime 側で既定が動いたときに
    // この家族だけ古い値で走る）。drafter Session にも同じノブを渡す。
    const sessionOptions = {
      ...(options.fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd: options.fuseRmsNormAdd }),
      ...(options.fuseLinearStaticQuantize === undefined
        ? {}
        : { fuseLinearStaticQuantize: options.fuseLinearStaticQuantize }),
      ...(options.rmsNormReduce === undefined ? {} : { rmsNormReduce: options.rmsNormReduce }),
      ...(options.submitPolicy === undefined ? {} : { submitPolicy: options.submitPolicy }),
      stateAttentionReduce: options.stateAttentionReduce ?? GEMMA4_STATE_ATTENTION_REDUCE,
      ...(options.linearGemvReduce === undefined
        ? {}
        : { linearGemvReduce: options.linearGemvReduce }),
      ...(options.planBackingBudgetBytes === undefined
        ? {}
        : { planBackingBudgetBytes: options.planBackingBudgetBytes }),
      ...(options.linearGemvRowsThreadTarget === undefined
        ? {}
        : { linearGemvRowsThreadTarget: options.linearGemvRowsThreadTarget }),
    };
    let session: Session | undefined;
    try {
      session = await admitted.component.createSession(gpu, sessionOptions);
      const greedyOutput = !gpu.gpuTimingEnabled && options.onRunDiagnostics === undefined
        ? createGemmaGreedyOutput(gpu, session, wiring.logits, wiring.vocabSize)
        : undefined;
      const drafter = await GemmaPipeline.#buildDrafter(
        admitted,
        session,
        gpu,
        sessionOptions,
      );
      return {
        gpu,
        ownsGpu,
        session,
        stateAttentionReduce: sessionOptions.stateAttentionReduce,
        ...(greedyOutput === undefined ? {} : { greedyOutput }),
        graph: admitted.component.graph,
        wiring,
        program: generationProgramFace(wiring),
        ple,
        tokenizer,
        config: admitted.config,
        ...(drafter === undefined ? {} : { drafter }),
        ...(speculativeK === undefined ? {} : { speculativeK }),
        // ゲートのノブは検査済み（`assertSpeculative`）の値をそのまま素通しする。
        ...(options.speculative?.gate === undefined
          ? {}
          : { speculativeGate: options.speculative.gate }),
        ...(options.planBackingBudgetBytes === undefined
          ? {}
          : { planBackingBudgetBytes: options.planBackingBudgetBytes }),
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      };
    } catch (error) {
      // 構築に失敗したら誰も解放できなくなるので、ここで返す。順序は借り手（drafter は
      // 張れていない = 借用計数は既に戻っている）→ 貸し手 → 内部で取った GPU。
      await disposeSteps([
        () => {
          if (session !== undefined) return session.dispose();
        },
        () => {
          if (ownsGpu) gpu.destroy();
        },
      ]).catch(() => {});
      throw error;
    }
  }

  /**
   * drafter Session を 1 本張る（投機を指定したときだけ）。
   *
   * 借りるのは target Session が既に GPU へ載せた埋め込み表で、`shared.tensor` →
   * 貸し手 initializer 名の対応は admission が確定させてある（`./speculative.ts`）。
   * バイトは 1 つも複製されない。
   */
  static async #buildDrafter(
    admitted: Gemma4Admission,
    target: Session,
    gpu: GpuContext,
    sessionOptions: SessionOptions,
  ): Promise<Gemma4Drafter | undefined> {
    const { drafter } = admitted;
    if (drafter === undefined) return undefined;
    let sharedWeights: Record<string, SharedWeight> = {};
    for (const borrower of Object.keys(drafter.admission.sharedWeights)) {
      sharedWeights = {
        ...sharedWeights,
        [borrower]: target.exportWeight(
          drafter.admission.sharedWeights[borrower],
        ),
      };
    }
    return {
      session: await drafter.component.createSession(gpu, {
        ...sessionOptions,
        sharedWeights,
      }),
      // 見積り専用（`estimateSessionMemory` が drafter の常駐重みをこれから引く）。
      graph: drafter.component.graph,
      outputs: drafter.admission.outputs,
      rope: admitted.config.rope,
      hiddenSize: drafter.admission.hiddenSize,
    };
  }

  /**
   * 会話 1 ターンを回し、**確定した文字列片**を流す（ADR 0084 決定 4 の逐次復号）。
   *
   * 会話の描画と符号化は `gemma4ChatPrompt`（射程外の入力はここで**同期に** fail loudly）、
   * 生成は `GenerationSequence` 1 本で、汲み切る / `break` する / 中断するのいずれでも
   * sequence は返る。**1 ターン = 1 sequence** なので過去 turn は残らない — 多ターンの会話を
   * 自分で回すなら {@link Gemma4Pipeline.sequence} を使う。
   *
   * 逐次表示が要らないなら {@link Gemma4ChatStream.text} で 1 本の文字列として受け取れる
   * （反復との併用は同期に throw する — 1 つのストリームは 1 通りにしか消費しない）。
   *
   * 停止条件は 2 層で、要求ごとに足せる（配布形の EOS 集合は常に効く）:
   * {@link Gemma4ChatOptions.stopTokens} は sequence 層（token id）、
   * {@link Gemma4ChatOptions.stopStrings} はこの層（復号後の本文）が判定する。
   *
   * drafter を載せた pipeline では**既定で投機を張る**（{@link Gemma4ChatOptions.speculative} で
   * ターンごとに切れる）。張ったターンは `done` の `speculation` に勘定が載る。配布形の推奨
   * sampler（温度 1）でも張り、本文は投機なしで回したときと同じ列になる。既定は自己採算ゲート
   * 付きなので、投機が負ける文脈では途中から decode 形へ落ちる（`speculation.plainSteps`）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（1 つの Session を 2 本の会話で同時に押さない）。
   */
  chat(
    messages: readonly Gemma4ChatMessage[],
    options: Gemma4ChatOptions,
  ): Gemma4ChatStream {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4Pipeline: dispose 済みでは生成できない");
    }
    // 受理集合は同期に落とす（GPU にも順番待ちにも入る前）。
    const prompt = gemma4ChatPrompt(this.#state.tokenizer, messages);
    const chosen = options.sampler ?? this.#state.config.sampler;
    // MUST: 要求は**発行時に写す**（ADR 0083 追記 2026-09-02）。本体（async generator）は最初の
    // `next()` まで走らないので、ここで `options` を読み切らないと `maxNewTokens` / `signal` は
    // 「汲み始めた時点の値」になる — 発行と消費の間に書き換えた option が黙って効く形である。
    // 配列と sampler 指定も**ここで**写す。受け手の複製（`GenerationSequence.generate` /
    // `createSampler`）は async generator の本体 = 最初の `next()` なので、それだけに任せると
    // 「発行してから汲み始めるまで」の窓で書き換えた停止集合・抽選指定が黙って効く。写し口は
    // `sampler.ts` の {@link snapshotSpec} 1 本を共有する（`Gemma4ChatSession.send` と同じ）。
    const sampler = chosen === undefined ? undefined : snapshotSpec(chosen);
    const request: GenerationRequest = {
      prompt,
      maxNewTokens: options.maxNewTokens,
      ...(options.stopTokens === undefined ? {} : { stopTokens: [...options.stopTokens] }),
      ...(sampler === undefined ? {} : { sampler }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    };
    // MUST: 受理集合の値域も**ここで**見る（検査の正本は `sequence.ts` の 1 本 — 式を写さない）。
    // `generate` は下の async generator の本体で呼ぶので、ここで呼ばないと `maxNewTokens: 0` の
    // ような要求が「汲み始めるまで落ちない」＝ 発行元から遠い場所でしか診断が出ない。
    assertGenerationRequestValues(this.#state.wiring.vocabSize, request);
    // 停止文字列の状態機械もここで作る（指定の検査と複製がその中で済む = 受理集合が同期に
    // 落ちる）。停止文字列が無ければ素通しになるので、経路を 2 本に割らない。
    const stopStrings = createStopStringFilter(options.stopStrings ?? []);
    const capacity = options.capacity;
    const onPrefill = options.onPrefill;
    const onToken = options.onToken;
    // 投機の DI と観測 hook も**発行時に**決める（本体は最初の `next()` まで走らない）。
    const speculative = speculativeSetup(this.#state, options.speculative);
    const onRun = runDiagnosticsHook(this.#state);

    let settle!: (stop: Gemma4ChatStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<Gemma4ChatStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // 二次的な通知路なので、読まれなくても unhandled rejection にしない。
    done.catch(() => {});

    const state = this.#state;
    const acquire = this.#acquire.bind(this);
    const disposed = (): boolean => this.#disposal !== undefined;
    const chunks = async function* (): AsyncGenerator<string, void, undefined> {
      let release: (() => void) | undefined;
      let sequence: GenerationSequence | undefined;
      let stream: GenerationStream | undefined;
      let failure: { readonly error: unknown } | undefined;
      /** 一致した停止文字列（この層の停止理由 — 立ったら sequence の消費もそこで止める）。 */
      let matched: string | undefined;
      try {
        // MUST: 本体の先頭でもう一度見る。async generator の本体は最初の `next()` まで走らない
        // ので、発行時の検査だけでは「発行 → dispose → 汲み始める」が抜ける。抜けた先でも
        // ランタイムが受け付けはしない（dispose 済み Session）が、真因から遠い**runtime の
        // 文言**で落ちるため、ここで**発行時と同じ pipeline の文言**へ揃える。
        if (disposed()) {
          throw new Error("Gemma4Pipeline: dispose 済みでは生成できない");
        }
        release = await acquire();
        sequence = await createGenerationSequence({
          session: state.greedyOutput?.session ?? state.session,
          ...(state.greedyOutput === undefined ? {} : { greedy: state.greedyOutput.greedy }),
          program: state.wiring,
          ...(capacity === undefined ? {} : { capacity }),
          ...(speculative === undefined ? {} : { speculative }),
          ...(onRun === undefined ? {} : { onRun }),
        });
        stream = sequence.generate(request);
        matched = yield* decodeChatChunks(
          stream,
          state.tokenizer.createDetokenizer(),
          stopStrings,
          onPrefill,
          onToken,
        );
      } catch (error) {
        failure = { error };
        // MUST: 包まずそのまま投げる（ADR 0083 決定 5 — 消費側が `error === signal.reason` で
        // 自分の中断を識別できる）。
        throw error;
      } finally {
        await completeChatTurn({
          where: "Gemma4Pipeline.chat",
          stream,
          matched,
          failure,
          settle,
          fail,
          release,
          cleanup: async () => {
            await sequence?.dispose();
          },
        });
      }
    };

    return chatStreamOf(
      closeableGenerator(chunks(), (failure) => {
        if (failure !== undefined) fail(failure.error);
        else settle({ reason: "closed", tokens: 0 });
      }),
      done,
    );
  }

  /**
   * 低レベル面 — 多ターンの会話を自分で回す（ADR 0083 決定 1〜4 の `GenerationSequence`）。
   *
   * `prompt` は token id 列なので、会話の描画は呼び手が通す（`./gemma` サブパス）— 最初の
   * ターンが {@link gemma4ChatPrompt}（`<bos>` 込みの全体）、2 ターン目以降は
   * `gemma4ChatTurn`（**その turn の差分だけ**。前 turn を閉じる `<turn|>` は
   * `GenerationSequence` の `pendingToken` が前置するので含めない — ADR 0083 決定 4）。
   * **返った実体は呼び手が `dispose()` する** — 会話が終わった時点で返すのが
   * 正で、取りこぼしても {@link Gemma4Pipeline.dispose} が巻き取る（Session を live な context
   * ごと畳まないため）。
   *
   * MUST: {@link Gemma4Pipeline.chat} との直列化はしない（別の会話は別の context なので
   * ランタイム側は受ける）。同時に走らせれば KV も 2 本ぶん常駐する。
   *
   * `capacity` はこの会話が確保する容量（省略時は配布形の既定）。KV の物理確保はここで済むので、
   * 短い会話に大きな容量を取らせない / 長い会話に必要なぶんだけ取る、の判断はこの 1 箇所である。
   *
   * drafter を載せた pipeline では**既定で投機を張る**（{@link Gemma4SequenceOptions.speculative}
   * で会話ごとに切れる）。借り手 context はこの sequence の寿命に束ねられ、`dispose()` が
   * 貸し手より先に畳む。自己採算ゲートの状態も sequence の寿命（ターンを跨いで測り続ける）。
   */
  async sequence(
    options: Gemma4SequenceOptions = {},
  ): Promise<GenerationSequence> {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4Pipeline: dispose 済みでは sequence を作れない");
    }
    const state = this.#state;
    const speculative = speculativeSetup(state, options.speculative);
    const onRun = runDiagnosticsHook(state);
    const inner = await createGenerationSequence({
      session: state.greedyOutput?.session ?? state.session,
      ...(state.greedyOutput === undefined ? {} : { greedy: state.greedyOutput.greedy }),
      program: state.wiring,
      ...(options.capacity === undefined ? {} : { capacity: options.capacity }),
      ...(speculative === undefined ? {} : { speculative }),
      ...(onRun === undefined ? {} : { onRun }),
    });
    // MUST: `await` 明けにもう一度見る。`dispose` の鎖本体は `#handed` を走査してから畳むので、
    // 確保の途中で dispose された実体は**どちらの経路からも畳まれない**（この再検査だけが
    // 塞げる窓 — runtime 側 `executor.ts` の `#createGenerationContext` と同型）。
    if (this.#disposal !== undefined) {
      await inner.dispose();
      throw new Error("Gemma4Pipeline: dispose 済みでは sequence を作れない");
    }
    // 正しく返された sequence は追跡から外す（外さないと、多ターン UI が会話ごとに作って
    // 畳んでも Set が単調増加し、`dispose` が破棄済みの実体を全数もう一度 await する）。
    // 実体そのものではなく薄い包みを渡すのは、`GenerationSequence` に pipeline を知らせる席を
    // 作らないため（生成面は最後までパイプライン非依存 — ADR 0083）。包みが持つのは**この
    // 追跡だけ**である（観測席は `onRun` hook として sequence の中へ降りたので、`generate` は
    // 素通しで足りる）。
    const handed: GenerationSequence = {
      capacity: inner.capacity,
      // 導出値なので包みも getter で素通しする（値を写すと「渡した瞬間の値」で固まる）。
      get used(): number {
        return inner.used;
      },
      generate: inner.generate,
      dispose: async (): Promise<void> => {
        await inner.dispose();
        // 失敗した dispose は外さない（context が返っていないので `dispose` が巻き取る側に残す）。
        this.#handed.delete(handed);
      },
    };
    this.#handed.add(handed);
    return handed;
  }

  /**
   * この pipeline で 1 会話を回すときの GPU メモリ必要量を見積もる（ADR 0070 決定 5 の estimator）。
   *
   * **ロードの後・sequence 生成の前**の面である。容量と chunk 長は実行時ノブなので、必要量は
   * 「呼び手が生成の形を決めた後」でなければ決まらない（ADR 0089 追記 2026-09-02 の据え置きと
   * 同じ読み — ロード面には結線しない）。
   *
   * MUST: 返るのは**必要側のカテゴリ別合計だけ**で、空き側との比較も可否判定もしない（同 決定 5）。
   * 判定の最終門は out-of-memory errorScope のままで、この見積りは事前診断である。
   *
   * NOTE: {@link Gemma4PipelineOptions.chunkBuckets} は見積りを動かさない（欄も持たない）—
   * 一時領域も入出力も attention の一時も物理行数 `M` に単調で、ピークは最大 `M` = prefill 形に
   * ある。バケットはその `M` より小さい形を足すだけである。バケット形の backing が decode 形と
   * **同時に常駐する**ぶんは、{@link Gemma4PipelineOptions.planBackingBudgetBytes} が
   * `peakAccountedBytes` の片側の項として上から押さえる（ADR 0095 決定 4）。
   *
   * 小出力decodeの補助Session・常駐入出力は `auxiliaryBytes` に別計上し、ピークにも加える。
   * 診断で小出力を無効にしたpipelineではこの欄を省略する（0）。
   *
   * NOTE: `AdmissionReport` は runtime の型で、`@karume/models` は再輸出しない（ADR 0008 の薄い面 —
   * 見積りを読む消費者は runtime の型をそのまま使う）。
   *
   * ## 投機（{@link Gemma4PipelineOptions.speculative} を渡した pipeline）
   *
   * 3 つが足される（ADR 0096 段 3 §3.3）:
   *
   * 1. **verify シナリオ** — `k+1` 行を 1 run で流す形（`scenarios` に `"verify"` として並ぶ）。
   *    行選択記号 `R` も `k+1` に束ねる（readback は選んだ行数ぶん要る）。
   * 2. **drafter の常駐重み** — `resident.weights` の各欄に足す。貸し手から借りている埋め込み表は
   *    runtime の常駐プランナが `shared` 席として除くので、二重に数えない。
   * 3. **借り手 context の state** — 借り物スロットは 1 バイトも確保しないので、増えるのは論理長
   *    uniform の 8 バイトだけである。
   *
   * drafter の run 1 本ぶん（入出力と中間）は `unaccounted` 側に置く — draft run は verify run と
   * 同時には走らないうえ、この形の必要量は `k` にも `capacity` にも依らない小さな定数である。
   */
  estimateSessionMemory(options: Gemma4EstimateOptions = {}): AdmissionReport {
    const { wiring, graph, gpu, drafter } = this.#state;
    const capacity = options.capacity ?? wiring.capacity;
    const chunkLength = assertChunkLength(
      options.chunkLength ?? wiring.chunkLength,
      this.#state.config,
    );
    if (!Number.isSafeInteger(capacity) || capacity < chunkLength) {
      throw new Error(
        `Gemma4Pipeline: capacity ${capacity} が chunkLength ${chunkLength} 未満`,
      );
    }
    if (capacity > wiring.maxPosition) {
      throw new Error(
        `Gemma4Pipeline: capacity ${capacity} が maxPosition ${wiring.maxPosition} を超えた`,
      );
    }
    // MUST: Session に渡したのと同じ予算を渡す（片方だけ既定に落ちると、報告のピークが実際の
    // 保持集合と別の予算を名乗る）。未指定は欄ごと渡さず runtime の既定に任せる。
    const budget = this.#state.planBackingBudgetBytes === undefined
      ? {}
      : { planBackingBudgetBytes: this.#state.planBackingBudgetBytes };
    // MUST: 渡す（states 形 attention のノード内一時は行ブロック枚数がこの上限だけで決まるので、
    // 省くと estimator が fail loudly する — 既定値で埋めない）。
    const maxStorageBufferBindingSize = gpu.limits.maxStorageBufferBindingSize;
    // verify の R は k+1・物理行数 M は sequence が流す形と同じ（バケットへ丸めた行数 —
    // `chunkLength: k+1` を名乗ると k < 3 で実 run より小さい形を見積る）。
    const rows = (this.#state.speculativeK ?? GEMMA4_DRAFT_STEPS) + 1;
    const verifyRows = physicalChunkRows(rows, wiring);
    const baseTarget = estimateGraphMemory(graph, planWeightResidency(graph), {
      // 行数記号 R は run では `last_row` の要素数が束縛する（入力 shape 由来）。見積りは入力を
      // 持たないので R = 1（通常の prefill / decode の形）を明示する。
      bindings: { [wiring.rowSymbol]: 1 },
      generation: {
        chunkLength,
        bindings: { [wiring.capacitySymbol]: capacity },
        // 投機の verify は `k+1` 行を 1 run で流す形（prefill / decode のどちらでもない）。
        ...(drafter === undefined ? {} : {
          scenarios: [{
            name: "verify",
            chunkLength: verifyRows,
            bindings: { [wiring.rowSymbol]: rows },
          }],
        }),
      },
      maxStorageBufferBindingSize,
      stateAttentionReduce: this.#state.stateAttentionReduce,
      ...budget,
    });
    const auxiliaryBytes = this.#state.greedyOutput?.extraBytes;
    const target = {
      ...baseTarget,
      ...(auxiliaryBytes === undefined ? {} : { auxiliaryBytes }),
      peakAccountedBytes: baseTarget.peakAccountedBytes + (auxiliaryBytes ?? 0),
    };
    if (drafter === undefined) return target;
    // 借り手ぶん（常駐重みと context の state）は drafter グラフを**同じ estimator に掛けて**
    // 引く（式をこの層で組み直さない — 借り物スロットと共有 initializer を外すのは runtime の
    // 常駐プランナと `GenerationContext.create` の分岐そのものである）。
    const borrower = estimateGraphMemory(
      drafter.graph,
      planWeightResidency(drafter.graph),
      {
        // 借り手 context は `chunkLength` 1 ちょうど・容量記号は貸し手から継承する。
        generation: {
          chunkLength: 1,
          bindings: { [wiring.capacitySymbol]: capacity },
        },
        maxStorageBufferBindingSize,
        ...budget,
      },
    );
    const weights = {
      compressedBytes: target.resident.weights.compressedBytes +
        borrower.resident.weights.compressedBytes,
      uncompressedBytes: target.resident.weights.uncompressedBytes +
        borrower.resident.weights.uncompressedBytes,
      expandedBytes: target.resident.weights.expandedBytes +
        borrower.resident.weights.expandedBytes,
      totalBytes: target.resident.weights.totalBytes + borrower.resident.weights.totalBytes,
    };
    const residentBytes = borrower.resident.weights.totalBytes + borrower.resident.stateBytes;
    return {
      resident: {
        weights,
        stateBytes: target.resident.stateBytes + borrower.resident.stateBytes,
      },
      // 形ごとの必要量は貸し手の 3 形（prefill / decode / verify）のまま — drafter の run は
      // 貸し手の run と同時には走らない（借り手 run は貸し手の run リースを取る）。
      scenarios: target.scenarios,
      planBackingBudgetBytes: target.planBackingBudgetBytes,
      ...(auxiliaryBytes === undefined ? {} : { auxiliaryBytes }),
      // 常駐は 2 つの Session が同時に抱えるので和。保持集合の上限（予算 vs 最大シナリオ）は
      // 貸し手の側がそのまま効く。
      peakAccountedBytes: target.peakAccountedBytes + residentBytes,
      unaccounted: [
        ...target.unaccounted,
        "drafter の run 1 本ぶんの入出力と中間（借り手 Session の slot backing — draft run は" +
        " verify run と同時に走らず、必要量は k にも capacity にも依らない）",
      ],
    };
  }

  /**
   * 静的配線の読み口（`sequence()` で回すときに chunk 長・位置上限・容量・語彙数・停止集合を
   * 読む）。**凍結**した値で、`stopTokens` も凍結コピーである。
   *
   * 出るのは数だけで、グラフ入力 / 出力の名前や `derivedInputs` は出ない（`GenerationProgram`
   * の doc — 配線の相手である Session が公開面に無いので読んでも使い道が無く、書ける口は
   * 「検証済み」という型の意味を壊す）。
   */
  get program(): GenerationProgram {
    return this.#state.program;
  }

  /** 資産から組んだ tokenizer（chat の描画・復号に要る — 同じ digest set の 1 員）。 */
  get tokenizer(): GemmaTokenizer {
    return this.#state.tokenizer;
  }

  /**
   * 配布形が宣言した sampler の**既定**（ADR 0083 決定 7 — 宣言が無ければ `undefined` = greedy）。
   *
   * MUST: 名前は「今この生成が使っている sampler」ではない。{@link Gemma4Pipeline.chat} は
   * 要求が `sampler` を省略したときだけこれを使い、要求が渡せばそちらが勝つ。低レベル面
   * （{@link Gemma4Pipeline.sequence}）を自分で回すときは `generate` の `sampler` へ**自分で
   * 渡す** — `GenerationSequence` は配布形を知らないので、渡さなければ低層の既定（温度 0）で
   * 走る（parity 門がその経路である）。
   */
  get defaultSampler(): Gemma4DefaultSampler | undefined {
    return this.#state.config.sampler;
  }

  /**
   * 解放する。渡した sequence を先に畳み、**drafter Session（居れば）→ target Session** の順に
   * 畳み、**内部で取得した GPU だけ**破棄し、最後に PLE sidecar のホストキャッシュを返す。
   *
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す。
   *
   * MUST: PLE も解放する。GPU 常駐と違い**ホスト RAM**（{@link Gemma4PipelineOptions.maxResidentPleBytes}
   * ぶん = 既定で最大 shard 2 本ぶん）なので、口が無いと「dispose 済みのハンドルを 1 つ持ち
   * 続ける」だけでその RAM がプロセス寿命まで残る。
   *
   * MUST: 途中の 1 本が投げても**残りの段まで進む**。`#disposal` は失敗も含めて 1 本を保持する
   * （2 度目も同じ拒否を返す = 再試行の口が無い）ので、最初の失敗で打ち切ると Session も GPU も
   * PLE のホスト RAM も**二度と**解放されない。失敗は 1 件ならそのまま、2 件以上は
   * `AggregateError` で運ぶ（どの段が落ちたかを消さない）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(async () => {
      const handed = [...this.#handed];
      this.#handed.clear();
      await disposeSteps([
        ...handed.map((sequence) => () => sequence.dispose()),
        // MUST: 借り手（drafter Session）を貸し手より先に畳む — 借り手が生きている間の
        // 貸し手 `dispose()` は runtime が fail loudly で断る（借り手の bind group が貸し手の
        // 重みバッファを掴んでいる）。順序を逆にすると、この 1 本が必ず失敗して残りの段は
        // 通るものの、報告に毎回「貸し出している」が載る。
        () => this.#state.greedyOutput?.dispose(),
        () => this.#state.drafter?.session.dispose(),
        () => this.#state.session.dispose(),
        () => {
          if (this.#state.ownsGpu) this.#state.gpu.destroy();
        },
        // 順序は GPU の後（走行中の生成は既に畳んであるので、ここで引き手はもう居ない）。
        () => this.#state.ple.dispose(),
      ]);
    });
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  /**
   * 直列化鎖の席を取り、返した関数で手放す。
   *
   * 席を取るのは**本体が回り出した時**（async generator の本体は最初の `next()` まで走らない）。
   * 発行時に取ると、汲まれないまま捨てられた stream が鎖を永久に握り、`dispose` まで
   * 巻き添えになる。
   */
  #acquire(): Promise<() => void> {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    return new Promise<() => void>((admitted) => {
      void this.#chain(() => {
        admitted(release);
        return held;
      });
    });
  }
}

/** 通常 Gemma 4。既存の取得・数値・MTP 契約を保つ入口。 */
export class Gemma4Pipeline extends GemmaPipeline {
  private constructor(state: Gemma4State) {
    super(state);
  }
  /**
   * 配布形から取得して組む（`loadManifest` → `resolveFiles` → **グラフ shard だけ**を
   * 取って `prepareModel` → 家族 admission → 重み shard と PLE sidecar の prefetch →
   * tokenizer と索引の取得 → 構築）。重み shard は Session を組むときに 1 本ずつ流れ、PLE
   * sidecar は**触った 1 本だけ**が永続キャッシュから読み直される（ADR 0070 / 0085 決定 3）。
   *
   * **`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST）。パッケージ版が検証した
   * 取得元は {@link GEMMA4_SOURCES}（`./config.ts`）の `"gemma4"` — 再現性を自分で固定するなら
   * `{ repo, revision }` を呼び手が明示する。文字列の `ref` は `{ repo }` と読む（= `main` 追従）。
   *
   * 手元の配布形は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）。HF の `owner/name` の綴りの門は通らず、network も CacheStorage も
   * 通らない（{@link Gemma4FromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: Gemma4FromPretrainedOptions = {},
  ): Promise<Gemma4Pipeline> {
    return new Gemma4Pipeline(
      await super.loadPretrained(GEMMA4_PIPELINE_NAME, ref, options),
    );
  }
  /**
   * 取得済み資産から組む。資産の解釈・グラフとの突合・id 空間の相互照合を全てここで済ませ、
   * **製品グラフの Session を 1 本張って**返す。
   *
   * 製品グラフは配布形の時点で常に分割されている（ADR 0081）ので、`model` は**宣言順の
   * shard 列**（先頭がグラフ shard）を受け、`fromPretrained` と同じ shard 逐次面へ流す
   * （受け口の実装は `src/hub/components.ts` — 7 家族共有の {@link assetComponentOpener}）。
   *
   * MUST: `config` は {@link fromPretrained} と**同じ門**（{@link parseGemma4PipelineConfig}）を
   * 通す。TS の型は未知キーも値域も見ないので、門が無いと `temperature: -1` のような宣言が
   * 3.7GiB を読み切った後の初 `chat` で初めて落ちる。宣言を検査するのは**バイト列を 1 本も
   * 開く前**である。
   */
  static async fromAssets(
    input: Gemma4Assets,
    options: Gemma4PipelineOptions = {},
  ): Promise<Gemma4Pipeline> {
    return new Gemma4Pipeline(
      await super.loadAssets(GEMMA4_PIPELINE_NAME, input, options),
    );
  }
}

/** 固定 mobile QAT の初期実装では drafter を受けない。 */
export type Gemma4QatPipelineOptions =
  & Omit<Gemma4PipelineOptions, "speculative">
  & { readonly speculative?: never };
export type Gemma4QatFromPretrainedOptions =
  & Omit<Gemma4FromPretrainedOptions, "speculative" | "model">
  & {
    readonly speculative?: never;
    readonly model?: "e2b" | "e4b";
  };

/** 実験段階の固定 QAT E2B/E4B。CPU/GPU の全ビット一致と広い品質は未保証（ADR 0097）。 */
export class Gemma4QatPipeline extends GemmaPipeline {
  private constructor(state: Gemma4State) {
    super(state);
  }
  /** 明示した QAT 配布形から E2B/E4B を組む。model 省略時は manifest の既定を使う。 */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: Gemma4QatFromPretrainedOptions = {},
  ): Promise<Gemma4QatPipeline> {
    return new Gemma4QatPipeline(
      await super.loadPretrained("gemma4-qat", ref, options),
    );
  }
  /** 取得済み固定 QAT 資産を検査して組む。モデルはグラフと PLE の構成から判別する。 */
  static async fromAssets(
    input: Gemma4Assets,
    options: Gemma4QatPipelineOptions = {},
  ): Promise<Gemma4QatPipeline> {
    return new Gemma4QatPipeline(
      await super.loadAssets("gemma4-qat", input, options),
    );
  }
}
