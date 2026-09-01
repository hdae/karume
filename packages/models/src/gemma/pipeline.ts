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

import { acquireGpu, type GpuContext, type Session } from "@karume/runtime";
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
  createGenerationSequence,
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
import { parseGemmaTokenizerAsset } from "./text/asset.ts";
import { GemmaTokenizer } from "./text/tokenizer.ts";
import { type Gemma4ChatMessage, gemma4ChatPrompt, gemma4StopTokens } from "./text/chat.ts";

/** グラフ入力の名前（正本は `export_product.py` の定数）。 */
const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

/**
 * 配布形（manifest）の取得キー — weights 1 本と、全量で受け取る assets 2 本。
 *
 * MUST: PLE sidecar の shard は {@link EAGER_ASSETS} に**入れない**。1 本 758MB 級で、全量常駐
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
   * — 中断が「この shard を読み終わってから」効くだけ）。1 本 758MB 級なので、対話的に止める
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
   * PLE sidecar shard の常駐本数（LRU — ADR 0085 決定 3。省略時は `ple.ts` の既定 2）。
   *
   * shard 1 本は 758MB 級なので、常駐を絞るほど RAM は減り、範囲をまたぐ会話では読み直しが
   * 増える。全部載せる（= shard 本数）と読み直しはゼロになる。
   */
  readonly residentPleShards?: number;
};

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

/** 1 ターンぶんの chat リクエスト。 */
export type Gemma4ChatOptions = {
  /** 生成する token 数の上限（1 以上）。停止 token はこの数に**含めない**。 */
  readonly maxNewTokens: number;
  /** sampling の指定（省略時は {@link Gemma4PipelineConfig.sampler}、それも無ければ greedy）。 */
  readonly sampler?: SamplerSpec;
  /** 中断（段の境目で検査し `signal.reason` をそのまま throw する — ADR 0083 決定 5）。 */
  readonly signal?: AbortSignal;
};

/**
 * 文字列片の列（`for await` で汲む）+ 停止理由。
 *
 * 片は逐次復号器が**確定させたぶん**だけで（ADR 0084 決定 4）、byte_fallback の途中は次の
 * token まで持ち越される。連結すると `decode(全 token id)` と一致する。
 *
 * MUST: `done` は**二次的な**通知路である（`GenerationStream.done` と同じ規律）— 失敗は
 * iterable 側が throw するのが一次で、`done` は同じ例外で reject するだけ。
 */
export type Gemma4ChatStream = AsyncIterable<string> & {
  readonly done: Promise<GenerationStop>;
};

/** {@link Gemma4Pipeline} の内部状態（公開面には出さない）。 */
type Gemma4State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly session: Session;
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
   * 解放口を持たないと shard 1 本 758MB 級 × 常駐 2 本がプロセス寿命まで残る。
   */
  readonly ple: Gemma4Ple;
  readonly tokenizer: GemmaTokenizer;
  readonly config: Gemma4PipelineConfig;
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
 * NOTE: `config` の検査はここには無い — 2 つの入口が**どちらも**
 * {@link parseGemma4PipelineConfig} を通してから呼ぶ（値域・関係・未知キーの門はそこが正本で、
 * 同じ検査を 2 実装持たない）。
 */
const admitGemma4 = (component: ModelComponent, config: Gemma4PipelineConfig): Gemma4Admission => {
  const { graph } = component;
  return {
    component,
    config,
    vocabSize: vocabSizeOf(graph),
    capacitySymbol: capacitySymbolOf(graph),
  };
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
    ...(options.residentPleShards === undefined
      ? {}
      : { residentShards: options.residentPleShards }),
  });

  const wiring = createGenerationProgram({
    graph: component.graph,
    inputIds: INPUT_IDS,
    positionIds: POSITION_IDS,
    lastRow: LAST_ROW,
    logits: component.graph.outputs[0],
    chunkLength: config.chunkLength,
    maxPosition: config.maxPosition,
    capacity: config.capacity,
    vocabSize,
    // 停止集合は tokenizer 資産の追加語彙から導出する（ADR 0083 決定 8 / 0084 決定 5 —
    // chat 形式と同じ digest set から来る）。
    stopTokens: gemma4StopTokens(tokenizer),
    bindings: { [capacitySymbol]: config.capacity },
    // ホスト由来の per-chunk 入力の席に PLE gather を差す（ADR 0085）。`options` は
    // そのまま降ろす — shard 1 本 758MB 級の読みが中断の届かない区間になるのを避ける。
    derivedInputs: {
      names: [PER_LAYER_INPUTS],
      derive: async (ids, deriveOptions) => ({
        [PER_LAYER_INPUTS]: await ple.gather(ids, deriveOptions),
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
        session: await admitted.component.createSession(gpu),
        wiring,
        program: generationProgramFace(wiring),
        ple,
        tokenizer,
        config: admitted.config,
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

    let settle!: (stop: GenerationStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<GenerationStop>((resolve, reject) => {
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
        });
        const detokenizer = state.tokenizer.createDetokenizer();
        stream = sequence.generate({
          prompt,
          maxNewTokens: options.maxNewTokens,
          ...(sampler === undefined ? {} : { sampler }),
          ...(options.signal === undefined ? {} : { signal: options.signal }),
        });
        for await (const event of stream) {
          if (event.kind !== "token") continue;
          const text = detokenizer.push(event.id);
          if (text !== "") yield text;
        }
        const tail = detokenizer.finish();
        if (tail !== "") yield tail;
      } catch (error) {
        failure = { error };
        // MUST: 包まずそのまま投げる（ADR 0083 決定 5 — 消費側が `error === signal.reason` で
        // 自分の中断を識別できる）。
        throw error;
      } finally {
        // 停止理由は**内側の `done` をそのまま**運ぶ（中断は resolve `aborted`・失敗は reject
        // という sequence 側の分け方を、ここで作り直さない）。
        if (stream === undefined) {
          if (failure !== undefined) fail(failure.error);
          else settle({ reason: "closed", tokens: 0 });
        } else {
          try {
            settle(await stream.done);
          } catch (error) {
            fail(error);
          }
        }
        // 1 ターン = 1 sequence（context を抱えたままにしない）。
        if (sequence !== undefined) await sequence.dispose();
        release?.();
      }
    };

    const iterable = chunks();
    return { [Symbol.asyncIterator]: () => iterable, done };
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
   */
  async sequence(): Promise<GenerationSequence> {
    if (this.#disposal !== undefined) {
      throw new Error("Gemma4Pipeline: dispose 済みでは sequence を作れない");
    }
    const inner = await createGenerationSequence({
      session: this.#state.session,
      program: this.#state.wiring,
    });
    // 正しく返された sequence は追跡から外す（外さないと、多ターン UI が会話ごとに作って
    // 畳んでも Set が単調増加し、`dispose` が破棄済みの実体を全数もう一度 await する）。
    // 実体そのものではなく薄い包みを渡すのは、`GenerationSequence` に pipeline を知らせる席を
    // 作らないため（生成面は最後までパイプライン非依存 — ADR 0083）。
    const handed: GenerationSequence = {
      // 導出値なので包みも getter で素通しする（値を写すと「渡した瞬間の値」で固まる）。
      get used(): number {
        return inner.used;
      },
      generate: (request) => inner.generate(request),
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
   * MUST: PLE も解放する。GPU 常駐と違い**ホスト RAM**（shard 1 本 758MB 級 × 既定 2 本）なので、
   * 口が無いと「dispose 済みのハンドルを 1 つ持ち続ける」だけで 1.5GiB がプロセス寿命まで残る。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(async () => {
      for (const sequence of this.#handed) await sequence.dispose();
      this.#handed.clear();
      await this.#state.session.dispose();
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
      // 順序は GPU の後（走行中の生成は既に畳んであるので、ここで引き手はもう居ない）。
      this.#state.ple.dispose();
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
