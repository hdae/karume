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

import {
  acquireGpu,
  type AdmissionReport,
  estimateGraphMemory,
  type GpuContext,
  planWeightResidency,
  type Session,
  type SessionDiagnostics,
  type StateAttentionReduce,
} from "@karume/runtime";
import {
  type AssetProgress,
  type CacheDiagnostic,
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
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
import { toManifestSource } from "../hub/repo-ref.ts";
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
  type GenerationSequence,
  type GenerationStop,
  type GenerationStream,
} from "../generation/sequence.ts";
import type { SamplerSpec } from "../generation/sampler.ts";
import {
  createGemma4Ple,
  type Gemma4Ple,
  type Gemma4PleReadOptions,
  parseGemma4PleIndex,
} from "./ple.ts";
import {
  GEMMA4_ROPE_LAYER_TYPES,
  GEMMA4_ROPE_PARTS,
  gemma4RopeInputName,
  gemma4RopeInputNames,
  gemma4RopeInputs,
} from "./rope.ts";
import {
  createStopStringFilter,
  type StopStringFilter,
  type StreamingDetokenizer,
} from "../text/detokenizer.ts";
import { parseGemmaTokenizerAsset } from "./text/asset.ts";
import { GemmaTokenizer } from "./text/tokenizer.ts";
import { type Gemma4ChatMessage, gemma4ChatPrompt, gemma4StopTokens } from "./text/chat.ts";

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
 * 配布形（manifest）の取得キー — weights 1 本と、全量で受け取る assets 2 本。
 *
 * MUST: PLE sidecar の shard は {@link EAGER_ASSETS} に**入れない**。1 本 250MiB 級 × 9 本で、全量常駐
 * させると ADR 0085 決定 3（触った shard だけ遅延ロード + LRU）そのものが成立しなくなる。
 * 取得キーは索引が書いたファイル名（`ple.json` の `shards[].file`）なので、遅延側の表は
 * 「eager に並べなかった残り」として自動的に PLE shard だけになる。
 */
const MODEL = "model";
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
   * PLE sidecar shard 1 本を取る（ファイル読み / hub の `streamAssets` — 呼び手の責務）。
   *
   * `options.signal` は**その読みを起こした生成**の中断で、**best-effort**（無視しても壊れない
   * — 中断が「この shard を読み終わってから」効くだけ）。1 本 250MiB 級なので、対話的に止める
   * 使い方をするなら見る価値がある。
   */
  readonly readPleShard: (file: string, options?: Gemma4PleReadOptions) => Promise<ArrayBuffer>;
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
   * 絞るほど RAM は減り、token 範囲をまたぐ会話では shard の読み直しが増える（値も token 列も
   * 変わらない）。`0` は「常駐させない」、sidecar 全体ぶんを渡せば読み直しはゼロになる。
   * 省略時は最大 shard 2 本ぶん（`ple.ts` の `defaultGemma4PleResidentBytes`）。
   *
   * NOTE: 本数ではなくバイトで受ける — shard 幅は資産世代で変わるので、「N 本」は世代ごとに
   * 違う RAM を意味する（ADR 0085 追記 2026-09-02）。
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
   * 実行 1 回ごとの診断を受け取る観測席（他 7 家族と同型）。op 別 GPU 時間（`lastRunTiming`）が
   * 要るときは `gpu` に `acquireGpu({ gpuTiming: true })` を渡す（ADR 0021 — 既定は計測しない）。
   *
   * 呼ばれるのは **run 1 本ごと**（prefill は chunk ごと・decode は step ごと）で、その run の
   * 完了後である。prefill 直後の最初の token は最終 chunk の logits から抽選するだけで run を
   * 伴わないので、そこでは呼ばない（呼ぶと同じ run の診断が 2 度届く）。
   *
   * MUST NOT: 届いた件数を走った run 数の代理に使わない。この席は生成イベント列に挟んで呼ぶ
   * ので、**停止 token を引いた最後の decode run** の診断は届かない（その run は `token` を
   * yield せずに終わる）。普通に喋り終わったターン（`eos` / `stop-token`）は必ずこの形なので、
   * 実運用では毎ターン 1 本欠ける。積算した GPU 時間も同じぶん過小になる。
   *
   * NOTE: 他ファミリと違ってコンポーネント名を渡さない — グラフが 1 本しかないので、名前が
   * 常に同じ 1 値になる（受け手が分岐できない引数を渡さない）。
   *
   * コールバックの例外は握らない（fail loudly — そのターンごと落ちる）。
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
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
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
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
 * {@link Gemma4Pipeline.fromPretrained} が追加で受けるもの（選択軸 + 取得層へ透過するノブ）。
 *
 * NOTE: `headers` / `fetch` / `caches` は **HTTP 取得元専用**のノブで、取得元ハンドル
 * （`localDirectory` / `denoDirectory`）を渡した呼び出しでは 1 つも効かない — 手元の配布形は
 * network も CacheStorage も通らない。
 */
export type Gemma4FromPretrainedOptions = Gemma4PipelineOptions & {
  /** manifest のモデル名（省略時は `defaultModel`）。 */
  readonly model?: string;
  /** quant 名（省略時はそのモデルの `defaultQuant`）。 */
  readonly quant?: string;
  /**
   * `Authorization` 等。取得（revision 解決・ファイル）へそのまま透過する。
   *
   * NOTE: **キャッシュは credential で分けない**（by-design — キーにヘッダは入らないので、
   * 認証付きで取得したバイト列は以後の無認証呼び出しにもヒットする）。ADR 0080 決定 3 /
   * `docs/limitations.md` の「hub: キャッシュは credential で隔離しない」が正本。
   */
  readonly headers?: HeadersInit;
  readonly onProgress?: (progress: AssetProgress) => void;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
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
  /** 中断（段の境目で検査し `signal.reason` をそのまま throw する — ADR 0083 決定 5）。 */
  readonly signal?: AbortSignal;
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
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly session: Session;
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
  /** 実行 1 回ごとの観測席（{@link Gemma4PipelineOptions.onRunDiagnostics}）。 */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
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
};

/**
 * 製品グラフ以外の資産（2 面が別の経路で用意し、解釈は 1 本に集める）。
 *
 * MUST: PLE sidecar だけ「バイト列」ではなく**読み口**を受ける（{@link Gemma4Assets} の同 MUST）。
 */
type Gemma4SidecarAssets = {
  readonly tokenizer: Uint8Array<ArrayBuffer>;
  readonly pleIndex: Uint8Array<ArrayBuffer>;
  readonly readPleShard: (file: string, options?: Gemma4PleReadOptions) => Promise<ArrayBuffer>;
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
 * 最終行 logits 出口の語彙数をグラフから引く（`[1, 1, V]` — ADR 0083 決定 6）。
 *
 * MUST: 呼び手に宣言させない。V は主 embedding の行数そのもので、宣言と食い違えば PLE
 * sidecar との相互照合（ADR 0085 決定 5）が**間違った基準**で通ってしまう。形の検査は
 * `createGenerationProgram` が同じ値でもう一度行う。
 */
const vocabSizeOf = (graph: GenerationGraph): number => {
  if (graph.outputs.length !== 1) {
    throw new Error(
      `Gemma4Pipeline: グラフ出力が ${graph.outputs.length} 本` +
        `（製品グラフの出口は最終行 logits の 1 本 — ADR 0083 決定 6）`,
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
      `Gemma4Pipeline: グラフ出力 '${name}' の shape [${shape.join(",")}] が [1,1,V] でない`,
    );
  }
  return vocab;
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
    for (const dim of input.shape) if (typeof dim === "string") fromInputs.add(dim);
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
const admitGemma4 = (component: ModelComponent, config: Gemma4PipelineConfig): Gemma4Admission => {
  const { graph } = component;
  assertRopeInputShapes(graph, config);
  return {
    component,
    config,
    vocabSize: vocabSizeOf(graph),
    capacitySymbol: capacitySymbolOf(graph),
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
 */
const assertChunkLength = (chunkLength: number, config: Gemma4PipelineConfig): number => {
  if (!Number.isSafeInteger(chunkLength) || chunkLength < 2) {
    throw new Error(`Gemma4Pipeline: chunkLength ${chunkLength} が 2 以上の整数でない`);
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
 * admission を通った材料 + 資産から静的配線を組む（`fromAssets` と `fromPretrained` が共有）。
 *
 * ここが id 空間の相互照合（ADR 0085 決定 5）を全部通す — ①tokenizer が生成しうる id
 * ②主 embedding の vocab 行数 ③PLE sidecar の行数。
 */
const buildGemma4Program = (
  admitted: Gemma4Admission,
  assets: Gemma4SidecarAssets,
  options: Gemma4PipelineOptions,
): {
  readonly wiring: GenerationWiring;
  readonly tokenizer: GemmaTokenizer;
  readonly ple: Gemma4Ple;
} => {
  const { config, vocabSize, capacitySymbol, component } = admitted;
  const tokenizer = new GemmaTokenizer(parseGemmaTokenizerAsset(assets.tokenizer));
  // ① tokenizer が生成しうる id と ② 主 embedding の vocab 行数。
  if (tokenizer.maxTokenId >= vocabSize) {
    throw new Error(
      `Gemma4Pipeline: tokenizer の最大 token id ${tokenizer.maxTokenId} が` +
        ` 主 embedding の vocab 行数 ${vocabSize} の外（別の語彙で焼かれた組み合わせ）`,
    );
  }
  // ③ PLE sidecar の行数（この突合は `createGemma4Ple` が持つ — 同じ検査を 2 実装持たない）。
  const ple = createGemma4Ple({
    index: parseGemma4PleIndex(
      JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(assets.pleIndex)),
    ),
    readShard: assets.readPleShard,
    vocabSize,
    ...(options.maxResidentPleBytes === undefined
      ? {}
      : { maxResidentBytes: options.maxResidentPleBytes }),
  });

  const wiring = createGenerationProgram({
    graph: component.graph,
    inputIds: INPUT_IDS,
    lastRow: LAST_ROW,
    logits: component.graph.outputs[0],
    chunkLength: assertChunkLength(options.chunkLength ?? config.chunkLength, config),
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
        ...gemma4RopeInputs(config.rope, positions),
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
): {
  readonly config: Gemma4PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
} => {
  const modelName = selection.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `Gemma4Pipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== GEMMA4_PIPELINE_NAME) {
    throw new Error(
      `Gemma4Pipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${GEMMA4_PIPELINE_NAME}/${GEMMA4_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== GEMMA4_PIPELINE_MAJOR) {
    throw new Error(
      `Gemma4Pipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${GEMMA4_PIPELINE_NAME}/${GEMMA4_PIPELINE_MAJOR}）`,
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
): AsyncGenerator<string, string | undefined, undefined> {
  for await (const event of events) {
    if (event.kind === "prefill") {
      // 文字列の面には prefill の片が無い（本文はまだ 1 文字も出ていない）ので、進捗だけを
      // 観測席へ渡す。例外は握らない（fail loudly — 呼び手のコールバックの誤りを飲まない）。
      onPrefill?.({ chunk: event.chunk, chunks: event.chunks });
      continue;
    }
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
 * 生成イベント列に観測席を挟む（{@link Gemma4PipelineOptions.onRunDiagnostics}）。
 *
 * 席が pipeline 層にあるのは、`GenerationSequence` が**パイプライン非依存**だからである（Session
 * も診断も知らない — ADR 0083）。イベントは run 1 本ごとに 1 通 …… ただし 1 箇所だけ例外があり、
 * prefill 直後の最初の token は「最終 chunk の logits から抽選しただけ」で run を伴わない。そこで
 * 呼ぶと同じ run の診断が 2 度届くので、最初の token だけ飛ばす（`tokens === 1`）。
 *
 * MUST: 観測席が無ければ**元の列をそのまま返す**（包みを 1 枚も増やさない — 中断や `return()` の
 * 伝播経路を、使わない人にまで足さない）。
 */
const withRunDiagnostics = (
  stream: GenerationStream,
  state: Pick<Gemma4State, "session" | "onRunDiagnostics">,
): GenerationStream => {
  const listener = state.onRunDiagnostics;
  if (listener === undefined) return stream;
  const events = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
    let tokens = 0;
    for await (const event of stream) {
      if (event.kind === "token") tokens += 1;
      if (event.kind !== "token" || tokens > 1) listener(state.session.diagnostics());
      yield event;
    }
  };
  const iterable = events();
  return { [Symbol.asyncIterator]: () => iterable, done: stream.done };
};

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
export class Gemma4Pipeline {
  readonly #state: Gemma4State;
  /** chat と dispose の直列化鎖（1 つの Session を 2 本の会話で同時に押さない）。 */
  readonly #chain = createOperationChain();
  /** {@link Gemma4Pipeline.sequence} が渡した実体（dispose の取りこぼしを塞ぐ）。 */
  readonly #handed = new Set<GenerationSequence>();
  /** dispose の 1 本。**undefined でないことが「dispose 済み」**（派生状態を別に持たない）。 */
  #disposal: Promise<void> | undefined;

  private constructor(state: Gemma4State) {
    this.#state = state;
  }

  /**
   * 配布形から取得して組む（`loadManifest` → `resolveFiles` → **グラフ shard だけ**を
   * 取って `prepareModel` → 家族 admission → 重み shard と PLE sidecar の prefetch →
   * tokenizer と索引の取得 → 構築）。重み shard は Session を組むときに 1 本ずつ流れ、PLE
   * sidecar は**触った 1 本だけ**が永続キャッシュから読み直される（ADR 0070 / 0085 決定 3）。
   *
   * **`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST）。gemma4 は公開配布
   * リポをまだ持たないので pin 定数も無い（`./config.ts` の NOTE）— `{ repo, revision }` を
   * 呼び手が明示する。文字列の `ref` は `{ repo }` と読む（= `main` 追従）。
   *
   * 手元の配布形は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）。HF の `owner/name` の綴りの門は通らず、network も CacheStorage も
   * 通らない（{@link Gemma4FromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: Gemma4FromPretrainedOptions = {},
  ): Promise<Gemma4Pipeline> {
    const where = "Gemma4Pipeline.fromPretrained";
    const source = toManifestSource(ref, where);
    const hubOptions: StreamAssetsOptions = {
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.headers === undefined ? {} : { headers: options.headers }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
    };
    const loaded = await loadManifest(source, hubOptions);
    const selection = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const files = resolveFiles(loaded.manifest, selection);
    const { admitted, assets, deferred } = await loadShardComponents(
      where,
      loaded,
      files,
      [MODEL],
      // 家族の門は admission 席で通す（重み shard を取る前 — `src/hub/components.ts`）。
      async (open) => {
        const { config, quantName, quant } = gemma4ManifestConfig(loaded.manifest, selection);
        const admitted = admitGemma4(open(MODEL), config);
        // 配布形が宣言した `requiredLimits` は**重み shard を取る前**にここで見る
        // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
        // 他 7 家族と違って席が閉包側にあるのは、{@link admitGemma4} が構築オプションを
        // 受け取らない（グラフだけで決まる）ため。
        await assertRequiredLimitsBeforeDownload(
          quant.requiredLimits,
          options.gpu,
          `Gemma4Pipeline: quant '${quantName}'`,
        );
        return admitted;
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
    const readPleShard = (
      file: string,
      readOptions: Gemma4PleReadOptions = {},
    ): Promise<ArrayBuffer> => {
      if (!Object.hasOwn(deferred, file)) {
        throw new Error(
          `${where}: PLE sidecar の shard '${file}' が manifest の assets に無い` +
            `（宣言されている shard: ${Object.keys(deferred).join(" / ")}）`,
        );
      }
      // MUST: 取得層のオプションから `signal` を落とす（`hub/components.ts` の相 2 と同じ理由 —
      // ロード 1 回の寿命を表す signal を、以後の生成が使う読み口へ持ち越さない）。載せ直すのは
      // **その読みを起こした生成**の signal だけで、寿命が読み 1 回と一致する。
      const { signal: _load, onProgress: _progress, ...streamOptions } = hubOptions;
      return readCachedAsset(where, loaded, deferred[file], {
        ...streamOptions,
        ...(readOptions.signal === undefined ? {} : { signal: readOptions.signal }),
      });
    };
    return await Gemma4Pipeline.#build(
      admitted,
      {
        tokenizer: assetBytes(where, assets, TOKENIZER_ASSET),
        pleIndex: assetBytes(where, assets, PLE_INDEX_ASSET),
        readPleShard,
      },
      options,
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
    const where = "Gemma4Pipeline.fromAssets";
    const config = parseGemma4PipelineConfig(input.config);
    if (input.model.length === 0) {
      throw new Error(`${where}: 製品グラフの shard 列が空（先頭がグラフ shard）`);
    }
    let shards: Record<string, Uint8Array<ArrayBuffer>> = {};
    input.model.forEach((bytes, index) => {
      shards = { ...shards, [`${MODEL}[${index}]`]: bytes };
    });
    const open = assetComponentOpener(where, shards, (key) => assetBuffer(where, shards, key));
    // NOTE: `requiredLimits` の検査はこの面には無い — {@link Gemma4Assets} は manifest を
    // 持たない（バイト列と `config` だけ）ので、宣言そのものへ到達できない。実寸の検査は
    // Session 構築時の `assertWeightsWithinLimits`（ADR 0089 決定 1）が受け持つ。
    const admitted = admitGemma4(open(MODEL), config);
    return await Gemma4Pipeline.#build(admitted, input, options);
  }

  /**
   * admission を通った材料 + 資産から組む（2 面が共有する 1 本）。
   *
   * MUST: 資産の解析は **GPU を取りに行く前**（他 7 家族と同じ順序）— 順序がずれると、GPU の
   * 無い環境では別の例外に化けて「何が悪かったのか」が読み手に伝わらない。
   *
   * Session を 1 本持ち続けるのは siglip2 と同じ理由で、畳む相手（同時に載せられない別の
   * 巨大グラフ）が居ないため — 会話ごとに張り直すと 1.5GiB の重みを毎回アップロードし直す。
   */
  static async #build(
    admitted: Gemma4Admission,
    assets: Gemma4SidecarAssets,
    options: Gemma4PipelineOptions,
  ): Promise<Gemma4Pipeline> {
    const { wiring, tokenizer, ple } = buildGemma4Program(admitted, assets, options);
    const gpu = options.gpu ?? await acquireGpu();
    const ownsGpu = options.gpu === undefined;
    try {
      return new Gemma4Pipeline({
        gpu,
        ownsGpu,
        // ③PV の縮約形は家族の既定（K-12 昇格済み）— 呼び手が明示すればそれに従う。
        session: await admitted.component.createSession(gpu, {
          stateAttentionReduce: options.stateAttentionReduce ?? GEMMA4_STATE_ATTENTION_REDUCE,
        }),
        graph: admitted.component.graph,
        wiring,
        program: generationProgramFace(wiring),
        ple,
        tokenizer,
        config: admitted.config,
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      });
    } catch (error) {
      // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
      if (ownsGpu) gpu.destroy();
      throw error;
    }
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
    const sampler = options.sampler ?? this.#state.config.sampler;
    // MUST: 要求は**発行時に写す**（ADR 0083 追記 2026-09-02）。本体（async generator）は最初の
    // `next()` まで走らないので、ここで `options` を読み切らないと `maxNewTokens` / `signal` は
    // 「汲み始めた時点の値」になる — 発行と消費の間に書き換えた option が黙って効く形である。
    // `prompt` と sampler 指定の複製は受け手の `GenerationSequence.generate` が済ませる。
    const request: GenerationRequest = {
      prompt,
      maxNewTokens: options.maxNewTokens,
      ...(options.stopTokens === undefined ? {} : { stopTokens: options.stopTokens }),
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
          session: state.session,
          program: state.wiring,
          ...(capacity === undefined ? {} : { capacity }),
        });
        stream = withRunDiagnostics(sequence.generate(request), state);
        matched = yield* decodeChatChunks(
          stream,
          state.tokenizer.createDetokenizer(),
          stopStrings,
          onPrefill,
        );
      } catch (error) {
        failure = { error };
        // MUST: 包まずそのまま投げる（ADR 0083 決定 5 — 消費側が `error === signal.reason` で
        // 自分の中断を識別できる）。
        throw error;
      } finally {
        // 停止理由は**内側の `done` をそのまま**運ぶ（中断は resolve `aborted`・失敗は reject
        // という sequence 側の分け方を、ここで作り直さない）。停止文字列だけはこの層の判定なので
        // 理由を差し替えるが、`tokens` は内側の数をそのまま使う（この層で数え直さない）。
        if (stream === undefined) {
          if (failure !== undefined) fail(failure.error);
          else settle({ reason: "closed", tokens: 0 });
        } else {
          try {
            const inner = await stream.done;
            // MUST: この層で起きた失敗（`onPrefill` / `onRunDiagnostics` / 復号器の未知 id・
            // 不正 UTF-8）は内側からは見えない — 内側は `return()` で閉じられて `closed` で
            // resolve するので、そのまま運ぶと `done` だけを読む呼び手が**失敗を成功として
            // 記録する**。中断は内側が `aborted` で運ぶ形が正なので、そこだけは触らない。
            if (failure !== undefined && inner.reason !== "aborted") fail(failure.error);
            else {
              settle(
                matched === undefined
                  ? inner
                  : { reason: "stop-string", stopString: matched, tokens: inner.tokens },
              );
            }
          } catch (error) {
            fail(error);
          }
        }
        // 1 ターン = 1 sequence（context を抱えたままにしない）。席は無条件に返す
        // （{@link closeChatTurn} の MUST）。
        await closeChatTurn("Gemma4Pipeline.chat", failure, async () => {
          if (sequence !== undefined) await sequence.dispose();
        }, release);
      }
    };

    return chatStreamOf(chunks(), done);
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
   */
  async sequence(options: Gemma4SequenceOptions = {}): Promise<GenerationSequence> {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4Pipeline: dispose 済みでは sequence を作れない");
    }
    const state = this.#state;
    const inner = await createGenerationSequence({
      session: state.session,
      program: state.wiring,
      ...(options.capacity === undefined ? {} : { capacity: options.capacity }),
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
    // 作らないため（生成面は最後までパイプライン非依存 — ADR 0083）。
    const handed: GenerationSequence = {
      capacity: inner.capacity,
      // 導出値なので包みも getter で素通しする（値を写すと「渡した瞬間の値」で固まる）。
      get used(): number {
        return inner.used;
      },
      // 観測席（家族固有）はここで挟む — 中の sequence は Session も診断も知らない。
      generate: (request) => withRunDiagnostics(inner.generate(request), state),
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
   * NOTE: `AdmissionReport` は runtime の型で、`@karume/models` は再輸出しない（ADR 0008 の薄い面 —
   * 見積りを読む消費者は runtime の型をそのまま使う）。
   */
  estimateSessionMemory(options: Gemma4EstimateOptions = {}): AdmissionReport {
    const { wiring, graph, gpu } = this.#state;
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
    return estimateGraphMemory(graph, planWeightResidency(graph), {
      bindings: {},
      generation: { chunkLength, bindings: { [wiring.capacitySymbol]: capacity } },
      // MUST: 渡す（states 形 attention のノード内一時は行ブロック枚数がこの上限だけで決まるので、
      // 省くと estimator が fail loudly する — 既定値で埋めない）。
      maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
    });
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
   * 解放する。渡した sequence を先に畳み、Session を畳み、**内部で取得した GPU だけ**破棄し、
   * 最後に PLE sidecar のホストキャッシュを返す。
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
      const failures: unknown[] = [];
      /** 後始末 1 段（失敗を集めて次の段へ進む）。 */
      const step = async (run: () => void | Promise<void>): Promise<void> => {
        try {
          await run();
        } catch (error) {
          failures.push(error);
        }
      };
      for (const sequence of this.#handed) await step(() => sequence.dispose());
      this.#handed.clear();
      await step(() => this.#state.session.dispose());
      await step(() => {
        if (this.#state.ownsGpu) this.#state.gpu.destroy();
      });
      // 順序は GPU の後（走行中の生成は既に畳んであるので、ここで引き手はもう居ない）。
      await step(() => this.#state.ple.dispose());
      if (failures.length === 1) throw failures[0];
      if (failures.length > 1) {
        throw new AggregateError(
          failures,
          `Gemma4Pipeline.dispose: 後始末が ${failures.length} 件失敗した`,
        );
      }
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
