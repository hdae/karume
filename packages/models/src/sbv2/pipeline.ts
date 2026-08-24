/**
 * `Sbv2Pipeline` — テキスト → 音声（SBV2 / Style-Bert-VITS2 JP-Extra）の 1 本の面。
 *
 * パイプライン（NN は全段 Karume・torch 不使用）:
 *
 * 1. テキスト解析（`@hdae/yomi` → 音素 / トーン / word2ph / DeBERTa トークン列。GPU 不要）
 * 2. `text_encoder`（DeBERTa）→ hidden_states の末尾から N 本目（N は `symbols.json`）
 * 3. ホスト: word2ph で BERT 特徴を音素レベルへ tile 展開
 * 4. ホスト: `style_vec` / `g` を配布形の表から行引き（`style.ts`）
 * 5. `front`（enc_p + dp + sdp reverse）→ **4 出力**（logw_sdp / logw_dp / m_p / logs_p）
 * 6. ホスト: 継続長 → フレーム展開 idx → z_p → 相対位置注意表
 * 7. `voice`（flow + dec 融合）→ f32 波形（WAV 化は `encodeWav` — パイプライン非依存の共通処理）
 *
 * ## MUST: グラフは 1 本ずつ開いて閉じる
 *
 * {@link Sbv2Pipeline.fromAssets} は **Session を 1 本も張らない** — 開くのはコンテナ
 * （`openModel` = ヘッダ解析のみ）までで、GPU 常駐は {@link Sbv2Pipeline.generate} の中で
 * 段ごとに張っては畳む。3 グラフ（text_encoder 334MB / front / voice）を同時に生かさない。
 *
 * MUST: この段取りは**公開 API 側でも**守る — `generate` は直列化鎖に載せ（並行呼び出しは
 * 待たされて順に走る）、`dispose` はその完了を待ってから GPU を破棄する。載せないと、並行
 * 呼び出し 2 本ぶんのグラフが同時常駐し、生成中の dispose が flush-before-destroy を破る。
 *
 * ## MUST: 低精度ノブが効くのは front / voice だけ
 *
 * quant の `session` は front / voice の Session にだけ渡す。配布形の `text_encoder` は
 * i8 の 1 dtype しか持たない（ADR 0026 が聴感ゲート込みで受理した 1 系列だけを配る）ので、
 * どの quant を選んでも同じ実体が来る — 実行形ノブの比較軸を SBV2 本体に保つ。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正は golden E2E（`packages/runtime/tests/e2e_sbv2_test.ts`）と、`tools/exporter/sbv2_demo.py
 * reference` による torch 突合（example の dump 経路）が担保する。
 *
 * ## NOTE: 日本語辞書だけが hub を経由しないネットワーク取得になっている
 *
 * ADR 0038 は資産の取得を hub に一元化しているが、`@hdae/yomi` の辞書（~19MB）は manifest の
 * weights / assets に無く、`fetchDictionaryBytes`（`@hdae/yomi/loader`）を**このパッケージが直接**
 * 呼んで取る。したがって `packages/models` が hub を経由せずネットワークへ出る唯一の経路で、
 * これは**テスト版として一旦この形**とするユーザー裁定（リファクタ時に再検討）。
 * 辞書を自前で管理したいホストとテストのために {@link Sbv2PipelineOptions.dictionary} の
 * 注入席を置いてあり、渡された場合は 1 度も取得に出ない。
 */

import {
  acquireGpu,
  createSession,
  type GpuContext,
  type KarumeModel,
  openModel,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type AssetProgress,
  type CacheDiagnostic,
  fetchAssets,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  resolveFiles,
} from "@karume/hub";
import { JtdDictionary, OverlayDictionary, type OverlayEntry } from "@hdae/yomi";
import { fetchDictionaryBytes } from "@hdae/yomi/loader";

import {
  parseSbv2PipelineConfig,
  SBV2_DEFAULT_SOURCE,
  SBV2_PIPELINE_MAJOR,
  SBV2_PIPELINE_NAME,
  type Sbv2PipelineConfig,
} from "./config.ts";
import {
  parseSbv2Table,
  type Sbv2Table,
  SPEAKER_TENSOR,
  speakerEmbedding,
  STYLE_TENSOR,
  styleVector,
} from "./style.ts";
import { Sbv2InputError } from "./errors.ts";
import { analyzeSbv2Text, type Sbv2TextAnalysis } from "./text/analyze.ts";
import type { Sbv2Prosody } from "./text/prosody.ts";
import { bertHiddenOutput, tileBertToPhoneLevel, type TiledBert } from "./text/bert-tile.ts";
import { buildRelPosTables } from "./text/rel-pos-tables.ts";
import { type JpExtraRules, parseJpExtraRules, type Sbv2Knobs } from "./text/symbols.ts";
import { type CleanRanges, DebertaTokenizer } from "./text/tokenizer.ts";
import { durationsToFrames } from "./host/duration.ts";
import { buildZp } from "./host/latent.ts";
import { Randn } from "./host/random.ts";
import { buildRelattnTables } from "./relattn-tables.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { toSessionOptions } from "../session/options.ts";

/**
 * manifest の weights / assets 表に現れる取得キー（ADR 0041 §3 の規約名）。
 * NOTE: `style_vectors` / `speaker_embeddings` は取得キーと safetensors のテンソル名が同綴り
 * （`style.ts` の {@link STYLE_TENSOR} / {@link SPEAKER_TENSOR}）だが、別の語彙なので別に持つ。
 */
const TEXT_ENCODER = "text_encoder";
const FRONT = "front";
const VOICE = "voice";
const TOKENIZER = "tokenizer";
const SYMBOLS = "symbols";
const STYLE_VECTORS = "style_vectors";
const SPEAKER_EMBEDDINGS = "speaker_embeddings";

/** 生成結果。`data` は f32 モノラル波形（`encodeWav` にそのまま渡せる）。 */
export type GeneratedAudio = {
  readonly sampleRate: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/**
 * テキスト解析の下書き（{@link Sbv2Pipeline.analyzeProsody} の返り値）。
 *
 * 編集して戻す面は {@link prosody}（句 / モーラ構造）で、`phones` / `tones` は**そこから
 * 導出した確認用**（`given_phone` / `given_tone` そのもの・add_blank 前・両端 PAD 込み）。
 *
 * MUST: 戻すのは `prosody` **だけ**（`generate({ text, prosody: draft.prosody })`）。派生欄まで
 * 受け取る形にすると、核を直して戻す往復で古い `tones` が同梱され、受け側は「無視する（編集を
 * 黙って捨てる）」か「落とす（正当な往復が通らない）」の二択になる。
 *
 * MUST: 公開するのはこの 3 欄だけ。word2ph・input_ids といった内部契約まで出すと、DeBERTa の
 * 語彙やトークナイザを差し替えた瞬間に外部が壊れる（それらは解析 1 回の中でしか整合しない）。
 */
export type Sbv2ProsodyDraft = {
  /** 編集対象（アクセント核を直して {@link Sbv2GenerateRequest.prosody} へ戻す）。 */
  readonly prosody: Sbv2Prosody;
  /** 音素列（両端 PAD 込み）。`prosody` からの派生・読み取り専用。 */
  readonly phones: readonly string[];
  /** トーン列（0/1、`phones` と同長）。`prosody` からの派生・読み取り専用。 */
  readonly tones: readonly number[];
};

/** 1 回の生成要求。未指定の欄は manifest の `pipelineConfig.defaults` が埋める。 */
export type Sbv2GenerateRequest = {
  readonly text: string;
  /** 話者名（`pipelineConfig.speakers` のキー）。 */
  readonly speaker?: string;
  /** スタイル名（`pipelineConfig.styles` のキー）。 */
  readonly style?: string;
  /** スタイルの強さ（1 = 選んだスタイルそのもの、0 = 平均スタイル）。 */
  readonly styleWeight?: number;
  /** sdp と dp の混合比（`logw = sdp·r + dp·(1−r)`）。 */
  readonly sdpRatio?: number;
  /** z_p のノイズ倍率。 */
  readonly noiseScale?: number;
  /** sdp reverse のノイズ倍率。 */
  readonly noiseScaleW?: number;
  /** 継続長のスケール（大きいほどゆっくり）。 */
  readonly lengthScale?: number;
  /** 乱数 seed（既定 0 — 同じ seed・同じノブなら同じ波形）。 */
  readonly seed?: number;
  /**
   * この 1 回だけ効かせる修正辞書。指定すると {@link Sbv2PipelineOptions.overlay} は**使わない**
   * （合成しない）— 1 回きりの語彙を既定の語彙へ足し込むと、次の合成にも効いたように見える。
   *
   * 解決済みの `OverlayDictionary` を渡せば毎回の解決を省ける（{@link Sbv2PipelineOptions.overlay}
   * の MUST を参照 — 同じ辞書に対して解決したものであることは検証できない）。
   */
  readonly overlay?: readonly OverlayEntry[] | OverlayDictionary;
  /**
   * 編集済みの下書き（{@link Sbv2Pipeline.analyzeProsody} で採った {@link Sbv2ProsodyDraft.prosody}
   * のアクセント核を直したもの）。
   *
   * MUST: 下書きは**同じ `text`・同じ `overlay`** で採ったものを戻す。解析が生む音素列と位置ごとに
   * 一致しなければ落とす（長さだけでなく内容まで見る）ので、text の取り違え・梱包規則の外部
   * 再実装のズレ・音素数が変わる編集（モーラの読み替え・記号の増減）はここで止まる。読みを
   * 変えたいときは {@link overlay} で解析からやり直す。
   *
   * MUST: {@link givenTone} との同時指定は落とす（どちらが勝つかの規則を作らない）。
   */
  readonly prosody?: Sbv2Prosody;
  /**
   * トーンの直接指定（`given_tone`）。**0/1 の生値**で、長さは解析の音素列
   * （add_blank 前・両端 PAD 込み = {@link Sbv2ProsodyDraft.phones} と同長）と一致する MUST。
   * 長さ・値域が外れたら落とす（黙って切り詰めない）。下書きは
   * {@link Sbv2Pipeline.analyzeProsody} で採る。
   *
   * 音素単位の低レベル席（梱包規則を呼び手が知っている前提）。アクセント句・モーラ単位で
   * 直すなら {@link prosody} を使う。
   */
  readonly givenTone?: readonly number[];
};

/** {@link Sbv2PipelineOptions.onRunDiagnostics} が受けるコンポーネント名（Session 1 本 = 1 名）。 */
export type Sbv2RunComponent = "text_encoder" | "front" | "voice";

/** 構築オプション（{@link Sbv2Pipeline.fromAssets} / {@link Sbv2Pipeline.fromPretrained} 共通）。 */
export type Sbv2PipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link Sbv2Pipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * `Session.run` 1 回ごとの診断を受け取る観測席（1 合成 = text_encoder → front → voice の
   * 3 回）。op 別 GPU 時間（`lastRunTiming`）が要るときは `gpu` に
   * `acquireGpu({ gpuTiming: true })` を渡す（ADR 0021 — 既定は計測しない）。
   * コールバックの例外は握らない（fail loudly — 合成ごと落ちる）。
   */
  readonly onRunDiagnostics?: (
    component: Sbv2RunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
  /**
   * 日本語辞書の注入席。省略すると初回の `generate` / `analyzeProsody` で
   * `fetchDictionaryBytes` が取りに出て、
   * 以降はインスタンスが保持する（モジュール doc の NOTE）。
   */
  readonly dictionary?: JtdDictionary;
  /**
   * 修正辞書（誤読・アクセントの差し替え）の既定席。このインスタンスの全ての解析に効く。
   * エントリ列を渡した場合の実体化は辞書が揃ってから 1 度だけ（以降は使い回す）。
   * 1 回きりの差し替えは {@link Sbv2GenerateRequest.overlay} を使う。
   *
   * 実行中に増減するユーザー辞書は、呼び手が `OverlayDictionary` を作り直して毎回の要求へ
   * 載せる運用になる（このインスタンスは可変状態を持たない）。解決済みのものを渡せば、
   * 数千語の辞書でも合成ごとの解決コストは掛からない。
   *
   * MUST: 解決済みで渡すなら**同じ辞書に対して解決したもの**であること。`OverlayDictionary` は
   * 構築元の辞書を保持しないので karume 側では検証できず、別の辞書で解決したものを渡すと
   * 文脈 ID がずれたまま合成が通る（沈黙誤値）。{@link dictionary} に `JtdDictionary` を渡して
   * 両側で 1 本を共有するのが確実。
   */
  readonly overlay?: readonly OverlayEntry[] | OverlayDictionary;
};

/** {@link Sbv2Pipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。 */
export type Sbv2FromPretrainedOptions = Sbv2PipelineOptions & {
  readonly signal?: AbortSignal;
  /** `Authorization` 等。付けた取得は認証専用のキャッシュ名前空間へ隔離される。 */
  readonly headers?: HeadersInit;
  readonly onProgress?: (progress: AssetProgress) => void;
  readonly onCacheError?: (diagnostic: CacheDiagnostic) => void;
  /** `fetch` の差し替え（テスト・カスタム輸送用）。 */
  readonly fetch?: typeof globalThis.fetch;
  /** `CacheStorage` の差し替え（テスト用）。 */
  readonly caches?: CacheStorage;
};

/** 取得済み資産から直接組むときの入力（hub の `fetchAssets` の返り値をそのまま渡す）。 */
export type Sbv2Assets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * 取得済みバイト列を `openModel` へ渡せる ArrayBuffer にする。
 *
 * MUST: `slice` で写さない — hub は buffer 全体を占める view を返す契約なので、崩れていたら
 * **取得層の不変条件破れ**として落とす。
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => {
  if (!Object.hasOwn(assets, key)) {
    throw new Error(
      `sbv2: 資産 '${key}' が無い（manifest の weights / assets に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `sbv2: 資産 '${key}' の bytes が buffer 全体を占めていない` +
        `（byteOffset ${bytes.byteOffset} / byteLength ${bytes.byteLength} /` +
        ` buffer ${bytes.buffer.byteLength}）`,
    );
  }
  return bytes.buffer;
};

/**
 * MUST: `fatal: true` で decode する。既定の TextDecoder は不正 UTF-8 を U+FFFD へ黙って
 * 置換するので、壊れたバイト列が「内容の違う valid JSON」として通ってしまう（hub の
 * manifest・anima tokenizer・safetensors ヘッダと同じ流儀で fail loudly）。
 *
 * NOTE: `export` は門を直接叩くテストのため（`fromAssets` 経由で此処へ届くには実 IR
 * コンテナ 3 本が要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const assetJson = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): unknown => {
  const buffer = assetBuffer(assets, key);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch (cause) {
    throw new Error(`sbv2: 資産 '${key}' が UTF-8 として読めない`, { cause });
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new Error(`sbv2: 資産 '${key}' が JSON として読めない`, { cause });
  }
};

/**
 * グラフ入力の 1 軸ぶんの**静的**次元を引く。
 *
 * MUST: パイプライン側に literal を置かない — コンテナが正で、モデルを差し替えたら値も追随する。
 * MUST: 形全体を静的と要求しない。`bert` は `[1, 1024, P]` で P が記号次元（音素数はデータ
 * 依存）なので、要る軸だけを見る。
 */
const staticInputDim = (model: KarumeModel, inputName: string, axis: number): number => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`グラフ入力 '${inputName}' が無い`);
  const dim = spec.shape[axis];
  if (typeof dim !== "number") {
    throw new Error(`グラフ入力 '${inputName}' の軸 ${axis} が静的次元でない（${String(dim)}）`);
  }
  return dim;
};

/** グラフ入力の軸 1（特徴幅）。`bert` の 1024・`style_vec` の 256・`g` の 512 を引く。 */
const featureWidth = (model: KarumeModel, inputName: string): number =>
  staticInputDim(model, inputName, 1);

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

const f32 = (data: Float32Array<ArrayBuffer>, shape: readonly number[]): Tensor => ({
  dtype: "f32",
  shape: [...shape],
  data,
});

const i32 = (values: readonly number[], shape: readonly number[]): Tensor => ({
  dtype: "i32",
  shape: [...shape],
  data: Int32Array.from(values),
});

const ones = (count: number): Float32Array<ArrayBuffer> => new Float32Array(count).fill(1);

/** グラフ出力を**位置**で引く（IR v1 の出力名は `output.<i>` — 名前を決め打ちしない）。 */
const outputAt = (
  model: KarumeModel,
  outputs: Readonly<Record<string, Tensor>>,
  index: number,
): Tensor => {
  const name = model.graph.outputs[index];
  if (name === undefined) {
    throw new Error(`グラフ出力 ${index} が無い（${model.graph.outputs.length} 本しかない）`);
  }
  const tensor = outputs[name];
  if (tensor === undefined) throw new Error(`グラフ出力 ${index}（'${name}'）が実行結果に無い`);
  return tensor;
};

/**
 * 1 グラフぶんの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の段が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 *
 * NOTE: `anima/pipeline.ts` にも同名の helper がある（**意図的な重複**）。あちらの `run` は
 * `graph.outputs[0]` を 1 本だけ返す形で、SBV2 の `front` は 4 出力（logw_sdp / logw_dp /
 * m_p / logs_p）なので載らない。Anima 側を多出力へ広げると全呼び出し側の分解が変わり、
 * 実 GPU でしか露見しない回帰リスクを負う — 共通化は両ファミリが揃ってからのリファクタに回す。
 */
const withSession = async <T>(
  gpu: GpuContext,
  model: KarumeModel,
  sessionOptions: SessionOptions,
  observe: ((diagnostics: SessionDiagnostics) => void) | undefined,
  body: (
    run: (inputs: Record<string, Tensor>) => Promise<Record<string, Tensor>>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  const session = await createSession(gpu, model, sessionOptions);
  try {
    const run = async (inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> => {
      const outputs = await session.run(inputs);
      if (observe !== undefined) observe(session.diagnostics());
      return outputs;
    };
    return await body(run, session);
  } finally {
    await session.dispose();
  }
};

/** 観測席（{@link Sbv2PipelineOptions.onRunDiagnostics}）へコンポーネント名を焼いて渡す。 */
const observer = (
  state: Sbv2State,
  component: Sbv2RunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/** Unicode コードポイントの上限（区間表はコードポイントの閉区間）。 */
const MAX_CODE_POINT = 0x10FFFF;

/**
 * `cleanRanges` の区間表を検査して読む。
 *
 * MUST: 整数・コードポイント範囲・`start <= end`・**昇順かつ非重複**まで見る。`inRanges`
 * （`text/tokenizer.ts`）は二分探索なので、この前提が破れても例外は出ず**黙って外す** —
 * 除去 / 空白化の規則だけが変わった `bertText` から `inputIds` と `baseWord2ph` が同じ
 * 壊れ方で作られるため、`text/analyze.ts` の長さ突合門も通り、別の BERT 埋め込みで合成した
 * 音がそのまま出る。並べ替えて救わない（資産の不正として構築時に落とす）。
 */
const parseRanges = (raw: unknown, where: string): (readonly [number, number])[] => {
  if (!Array.isArray(raw)) throw new Error(`${where}: 区間表が配列でない`);
  let previousEnd = -1;
  return raw.map((entry, index) => {
    if (
      !Array.isArray(entry) || entry.length !== 2 ||
      typeof entry[0] !== "number" || typeof entry[1] !== "number"
    ) {
      throw new Error(`${where}[${index}]: 区間が [start, end] の数値対でない`);
    }
    const [start, end]: [number, number] = [entry[0], entry[1]];
    if (
      !Number.isInteger(start) || !Number.isInteger(end) ||
      start < 0 || end > MAX_CODE_POINT
    ) {
      throw new Error(
        `${where}[${index}]: 区間 [${start}, ${end}] が 0..${MAX_CODE_POINT} の整数対でない`,
      );
    }
    if (start > end) {
      throw new Error(`${where}[${index}]: 区間 [${start}, ${end}] の start が end より大きい`);
    }
    if (start <= previousEnd) {
      throw new Error(
        `${where}[${index}]: 区間 [${start}, ${end}] が直前の終端 ${previousEnd} 以下から始まる` +
          "（昇順・非重複でない — 二分探索が黙って外す）",
      );
    }
    previousEnd = end;
    return [start, end] as const;
  });
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** i32 の上限（トークン id は最終的に `Int32Array` へ書かれる）。 */
const MAX_TOKEN_ID = 2147483647;

/**
 * 実行時資産のトークナイザ JSON（`karume dist` が配る形）を検査して読む。
 *
 * MUST: 構造を検査してから使う。壊れた語彙表は「読めない」ではなく**全トークンが `[UNK]`**
 * という形で沈黙し、BERT 特徴だけが静かに無意味になる（`text/tokenizer.ts` の doc）。
 *
 * NOTE: `export` は門を直接叩くテストのため（{@link assetJson} と同じ理由）。
 */
export const parseTokenizerAsset = (raw: unknown, where: string): DebertaTokenizer => {
  if (!isRecord(raw)) throw new Error(`${where}: オブジェクトでない`);
  const special = raw["special"];
  if (!isRecord(special)) throw new Error(`${where}.special: オブジェクトでない`);
  const [clsId, sepId, unkId] = [special["clsId"], special["sepId"], special["unkId"]];
  if (typeof clsId !== "number" || typeof sepId !== "number" || typeof unkId !== "number") {
    throw new Error(`${where}.special: clsId / sepId / unkId が数値でない`);
  }
  // MUST: 整数かつ i32 の範囲。id は `Int32Array` へ書かれるので、非整数は**黙って切り捨て
  // られ**、範囲外は wrap する — どちらも「別のトークンを指す」沈黙誤値になる（語彙の行数に
  // 収まることは `fromVocabText` が語彙表を持つ側で見る）。
  for (const [name, id] of [["clsId", clsId], ["sepId", sepId], ["unkId", unkId]] as const) {
    if (!Number.isInteger(id) || id < 0 || id > MAX_TOKEN_ID) {
      throw new Error(`${where}.special.${name}: 0..${MAX_TOKEN_ID} の整数でない（${id}）`);
    }
  }
  const vocabText = raw["vocabText"];
  if (typeof vocabText !== "string" || vocabText.length === 0) {
    throw new Error(`${where}.vocabText: 空`);
  }
  const clean = raw["cleanRanges"];
  if (!isRecord(clean)) throw new Error(`${where}.cleanRanges: オブジェクトでない`);
  const ranges: CleanRanges = {
    removed: parseRanges(clean["removed"], `${where}.cleanRanges.removed`),
    spaced: parseRanges(clean["spaced"], `${where}.cleanRanges.spaced`),
  };
  return DebertaTokenizer.fromVocabText(vocabText, ranges, { clsId, sepId, unkId });
};

/**
 * 配布形の表（`style_vectors` / `speaker_embeddings`）が名前表とグラフ入力の両方に合うことを
 * 構築時に見る。
 *
 * MUST: 行数の突合を落とさない。行番号は表の物理行そのものなので、名前表と表の行数がずれても
 * shape は合ったまま**別のスタイル・別の話者の声が出る**（`style.ts` の doc）。
 */
const assertTableFits = (
  table: Sbv2Table,
  names: ReadonlyMap<string, number>,
  key: string,
  cols: number,
): void => {
  if (table.rows !== names.size) {
    throw new Error(
      `Sbv2Pipeline: 資産 '${key}' の行数 ${table.rows} が pipelineConfig の ${names.size} 件` +
        `（${[...names.keys()].join(" / ")}）と違う — 行番号は表の物理行そのもの`,
    );
  }
  if (table.cols !== cols) {
    throw new Error(
      `Sbv2Pipeline: 資産 '${key}' の列数 ${table.cols} がグラフ入力の ${cols} と違う`,
    );
  }
};

/**
 * {@link Sbv2Pipeline} の内部状態。
 *
 * NOTE: 公開面（`mod.ts` / `sbv2.ts`）には出さない。torch 参照突合の dump 経路
 * （`examples/sbv2/dump.ts`）が {@link synthesizeSbv2} を直に呼ぶため関数だけを `export` して
 * いる — チェーンの実装を 2 本持たずに中間値を観測するための唯一の口で、利用者ストーリーでは
 * ない。この型自体は呼び出し側で推論に載るので、モジュール内に閉じる。
 */
type Sbv2State = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: Sbv2PipelineConfig;
  readonly rules: JpExtraRules;
  readonly tokenizer: DebertaTokenizer;
  readonly styles: Sbv2Table;
  readonly speakers: Sbv2Table;
  readonly textEncoder: KarumeModel;
  readonly front: KarumeModel;
  readonly voice: KarumeModel;
  readonly sessionOptions: SessionOptions;
  readonly onRunDiagnostics?: (
    component: Sbv2RunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
  /** 既定席の修正辞書（{@link Sbv2PipelineOptions.overlay}）。生のエントリ列か解決済みか。 */
  readonly defaultOverlay: readonly OverlayEntry[] | OverlayDictionary | undefined;
  /**
   * 日本語辞書の取得（可変欄 — インスタンス状態）。**値ではなく Promise** を持つ
   * （{@link ensureDictionary} の MUST）。
   */
  dictionary: Promise<JtdDictionary> | undefined;
  /**
   * `defaultOverlay` をエントリ列として辞書に対して解決したもの（可変欄）。辞書が揃うまで
   * 作れないので遅延で埋める。**解決は (辞書, entries) の純関数**で、両者ともここから先は
   * 不変なので、値が独立に動く余地は無い（1 度だけ作って持つ）。
   */
  defaultOverlayResolved: OverlayDictionary | undefined;
};

/** 辞書バイト列の取得（`fetchDictionaryBytes` と同型 — テストが差し替える席）。 */
export type DictionaryLoader = () => Promise<Uint8Array>;

/**
 * 日本語辞書を確保する（初回だけ取得に出て、以降はインスタンスが持つ — モジュール doc の NOTE）。
 * `generate` も `analyzeProsody` もここを通るので、19MB を取りに出るのは 1 度きり。
 *
 * MUST: 保持するのは**値ではなく Promise**。値で持つと「1 度きり」を直列化鎖でしか担保できず、
 * GPU を使わない解析まで進行中の合成の後ろに並ぶ。取得を先に置けば、await 中に並行で入った
 * 呼び出しも同じ取得へ合流する（鎖なしで 1 度きりが成り立つ）。
 *
 * MUST: 失敗した Promise は欄から捨てる。持ち続けると、1 度のネットワーク失敗が以後の全ての
 * 合成・解析へ波及し続ける（値キャッシュには無かった失敗モード）。
 *
 * NOTE: `load` の差し替え席と `export` はテストのため（実取得は 19MB を取りに出る）。
 */
export const ensureDictionary = (
  state: Pick<Sbv2State, "dictionary">,
  load: DictionaryLoader = fetchDictionaryBytes,
): Promise<JtdDictionary> => {
  state.dictionary ??= load()
    .then((bytes) => JtdDictionary.load(new Uint8Array(bytes).buffer))
    .catch((error: unknown) => {
      state.dictionary = undefined;
      throw error;
    });
  return state.dictionary;
};

/**
 * 生のエントリ列なら辞書に対して解決する（解決済みならそのまま使う）。
 *
 * MUST: 解決の失敗は {@link Sbv2InputError} へ包み直す。`overlay` は要求ごとに呼び手が渡す
 * 入力（ユーザー辞書 UI 由来の表記ゆれ・アクセント型の範囲外はごく普通の 400）で、yomi が
 * 投げる素の `Error` のままだと内部不変条件の破れ（500）と区別が付かない。`cause` を繋ぐので
 * メッセージの質は落ちない（errors.ts の MUST）。
 */
const resolveOverlay = (
  dictionary: JtdDictionary,
  overlay: readonly OverlayEntry[] | OverlayDictionary,
): OverlayDictionary => {
  if (overlay instanceof OverlayDictionary) return overlay;
  try {
    return new OverlayDictionary(dictionary, overlay);
  } catch (cause) {
    throw new Sbv2InputError(
      `Sbv2Pipeline: overlay の修正辞書エントリを解決できない（${
        cause instanceof Error ? cause.message : String(cause)
      }）`,
      { cause },
    );
  }
};

/**
 * この 1 回の解析に効かせる修正辞書を決める。
 *
 * MUST: 要求側の指定は既定席と**合成せず置き換える**（{@link Sbv2GenerateRequest.overlay}）。
 * 足し込むと、1 回きりのつもりで渡した語彙が「既定にも入っていた」ように見え、次の合成で
 * 消えたのか残ったのかを呼び出し側が判別できない。
 *
 * NOTE: 既定席のキャッシュ（`defaultOverlayResolved`）は鎖の外からも並行で埋まり得るが、解決は
 * (辞書, entries) の純関数なので、競り合っても二重構築が起きるだけで値は同値（先に返した側も
 * 同じ辞書を使い続ける）。
 *
 * NOTE: 受けるのは状態の 2 欄だけ（`export` は門を直接叩くテストのため — 合成経路でここへ
 * 届くには実 IR コンテナ 3 本が要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const overlayFor = (
  cache: Pick<Sbv2State, "defaultOverlay" | "defaultOverlayResolved">,
  dictionary: JtdDictionary,
  requested: readonly OverlayEntry[] | OverlayDictionary | undefined,
): OverlayDictionary | undefined => {
  if (requested !== undefined) return resolveOverlay(dictionary, requested);
  if (cache.defaultOverlay === undefined) return undefined;
  cache.defaultOverlayResolved ??= resolveOverlay(dictionary, cache.defaultOverlay);
  return cache.defaultOverlayResolved;
};

/** 合成 1 回ぶんの中間値（dump 経路と診断だけが読む）。 */
type Sbv2Trace = {
  readonly analysis: Sbv2TextAnalysis;
  readonly speaker: string;
  readonly style: string;
  readonly styleWeight: number;
  readonly knobs: Sbv2Knobs;
  readonly seed: number;
  /** 実際に採った hidden_states のグラフ出力名。 */
  readonly bertHiddenOutput: string;
  readonly xMask: Float32Array<ArrayBuffer>;
  readonly zNoise: Float32Array<ArrayBuffer>;
  readonly wCeil: Int32Array<ArrayBuffer>;
  readonly zpNoise: Float32Array<ArrayBuffer>;
  readonly channels: number;
  readonly frames: number;
};

/** {@link synthesizeSbv2} の返り値。{@link Sbv2Pipeline.generate} はここから音だけを採る。 */
type Sbv2Synthesis = {
  readonly sampleRate: number;
  readonly audio: Float32Array<ArrayBuffer>;
  readonly trace: Sbv2Trace;
};

const resolveRow = (
  names: ReadonlyMap<string, number>,
  name: string,
  where: string,
): number => {
  const row = names.get(name);
  if (row === undefined) {
    throw new Sbv2InputError(
      `${where} '${name}' は manifest に無い（利用可能: ${[...names.keys()].join(" / ")}）`,
    );
  }
  return row;
};

const finiteKnob = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new Sbv2InputError(`${name} ${value} が有限の数でない`);
  return value;
};

/**
 * トークン列の運用上限（`pipelineConfig.maxTokens`）を見る門。
 *
 * MUST: 相対位置表（`c2p_pos` / `p2c_pos` は各 4·T² bytes）の生成と Session の確保**より前**に
 * 呼ぶ。上限は焼いたグラフの記号次元の上限そのもので、超えた要求はどう確保しても実行できない。
 *
 * NOTE: `export` は門を直接叩くテストのため（合成経路でここへ届くには日本語辞書と実 IR
 * コンテナ 3 本が要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
 */
export const assertTokenLimit = (tokens: number, maxTokens: number): void => {
  if (tokens > maxTokens) {
    throw new Sbv2InputError(
      `Sbv2Pipeline: トークン数 ${tokens} が配布形の上限 maxTokens=${maxTokens} を超えている` +
        "（text を短く分けて合成する）",
    );
  }
};

/**
 * 音素列（front の記号次元 P）の運用上限を見る門。
 *
 * MUST: {@link assertTokenLimit} と**別に**要る。P = 2·sum(baseWord2ph)+1 で、base の各要素は
 * 1 以上・その本数が T なので **P >= 2·T+1 > T** が常に成り立つ。T 側の門だけでは P の超過を
 * 一度も捕まえられない（T <= 上限のまま P > 上限になる長文が実在する）。
 * MUST: DeBERTa を回す**前**に呼ぶ。後ろに置くと、超過が確定している要求に対して
 * text_encoder（334MB）を張って回し切ってから front の記号次元で落ちる。
 *
 * NOTE: 上限は T と P で同じ 1 つ（`pipelineConfig.maxTokens`）— 焼いた記号次元の上限が
 * 1 組であることは配布側が固定している（`tools/export-recipes/sbv2/tests/test_distribution.py`
 * の `the_limits_are_the_symbolic_maxima_the_export_scripts_baked`）ので、欄は増やさない。
 * NOTE: `export` は門を直接叩くテストのため（{@link assertTokenLimit} と同じ事情）。
 */
export const assertPhonemeLimit = (phonemes: number, maxTokens: number): void => {
  if (phonemes > maxTokens) {
    throw new Sbv2InputError(
      `Sbv2Pipeline: 音素数 ${phonemes} が配布形の上限 maxTokens=${maxTokens} を超えている` +
        "（text を短く分けて合成する）",
    );
  }
};

/**
 * tile 展開の結果を front へ渡す前に検める門。
 *
 * MUST: **書いた列数**（{@link TiledBert.columns}）を確保サイズと記号次元の両方に突き合わせる。
 * 「展開長 === 音素数」の 1 本だけでは、両辺が同じ `analysis` から導かれる定理になって恒真
 * （`sum(word2ph) === P` は `buildBaseWord2ph` が既に強制済み）で、走査が壊れても緑のまま通る。
 *
 * NOTE: `export` は門を直接叩くテストのため（{@link assertTokenLimit} と同じ事情）。
 */
export const assertTiledBert = (tiled: TiledBert, phonemes: number): void => {
  if (tiled.data.length !== tiled.dim * tiled.columns) {
    throw new Error(
      `BERT 展開が書いた ${tiled.columns} 列 × dim ${tiled.dim} が確保した` +
        ` ${tiled.data.length} 要素と合わない（tile 走査の破れ）`,
    );
  }
  if (tiled.columns !== phonemes) {
    throw new Error(`BERT 展開長 ${tiled.columns} が音素数 ${phonemes} と違う`);
  }
};

/**
 * manifest + 資産から実行状態を組む（{@link Sbv2Pipeline.fromAssets} の中身）。
 *
 * MUST: manifest の契約違反と**資産の解析・表の突合**は **GPU を取りに行く前**に落とす。
 * 順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に
 * 伝わらない。GPU 取得後に許される検査は GPU の能力（shader-f16）だけ。
 * MUST: Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
 */
export const openSbv2State = async (
  input: Sbv2Assets,
  options: Sbv2PipelineOptions = {},
): Promise<Sbv2State> => {
  const { manifest, assets } = input;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `Sbv2Pipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== SBV2_PIPELINE_NAME) {
    throw new Error(
      `Sbv2Pipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${SBV2_PIPELINE_NAME}/${SBV2_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== SBV2_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `Sbv2Pipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${SBV2_PIPELINE_NAME}/${SBV2_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseSbv2PipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `Sbv2Pipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];
  const wantsShaderF16 = quant.gpuFeatures?.shaderF16 === true;

  const front = openModel(assetBuffer(assets, FRONT));
  const voice = openModel(assetBuffer(assets, VOICE));
  const textEncoder = openModel(assetBuffer(assets, TEXT_ENCODER));
  const rules = parseJpExtraRules(assetJson(assets, SYMBOLS), SYMBOLS);
  const tokenizer = parseTokenizerAsset(assetJson(assets, TOKENIZER), TOKENIZER);
  const styles = parseSbv2Table(assetBuffer(assets, STYLE_VECTORS), STYLE_TENSOR);
  const speakers = parseSbv2Table(assetBuffer(assets, SPEAKER_EMBEDDINGS), SPEAKER_TENSOR);
  assertTableFits(styles, config.styles, STYLE_VECTORS, featureWidth(front, "style_vec"));
  assertTableFits(speakers, config.speakers, SPEAKER_EMBEDDINGS, featureWidth(front, "g"));
  // front と voice は同じ `g` を受ける。幅が割れていたら片方だけ別の話者表を要求している。
  const voiceGin = featureWidth(voice, "g");
  if (voiceGin !== speakers.cols) {
    throw new Error(
      `Sbv2Pipeline: voice のグラフ入力 'g' の幅 ${voiceGin} が話者表の列数 ${speakers.cols}` +
        " と違う（front / voice が別の話者埋め込みを要求している）",
    );
  }
  // 低精度ノブは front / voice の Session にだけ効かせる（モジュール doc の MUST）。
  const sessionOptions = toSessionOptions(quant.session);

  // MUST: `shader-f16` は device 作成時にしか要求できない（ADR 0028）。共有 GPU を渡された
  // 場合は要求できないので、能力が足りないことを**ここで**名指しして落とす。
  const gpu = options.gpu ?? await acquireGpu(wantsShaderF16 ? { shaderF16: true } : {});
  const ownsGpu = options.gpu === undefined;
  if (wantsShaderF16 && !gpu.shaderF16Enabled) {
    // 内部で取った GPU は、ここで投げると誰も解放できなくなるので返してから落とす。
    if (ownsGpu) gpu.destroy();
    throw new Error(
      `Sbv2Pipeline: quant '${quantName}' は shader-f16 を要求するが、渡された` +
        " GpuContext で有効になっていない（acquireGpu({ shaderF16: true }) で取り直す）",
    );
  }
  return {
    gpu,
    ownsGpu,
    config,
    rules,
    tokenizer,
    styles,
    speakers,
    textEncoder,
    front,
    voice,
    sessionOptions,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
    defaultOverlay: options.overlay,
    // 注入された辞書は「取得済みの Promise」として持つ（欄の型を 1 本にして分岐を作らない）。
    dictionary: options.dictionary === undefined ? undefined : Promise.resolve(options.dictionary),
    defaultOverlayResolved: undefined,
  };
};

/** 内部で取得した GPU だけを破棄する（渡された GpuContext は呼び出し側の所有物）。 */
export const closeSbv2State = (state: Sbv2State): void => {
  if (state.ownsGpu) state.gpu.destroy();
};

/**
 * テキスト 1 本を波形にする（中間値つき）。{@link Sbv2Pipeline.generate} はこれを呼んで
 * 音だけを返す。
 *
 * MUST: 乱数の消費順は `z_noise`（front の前）→ `zp_noise`（front の後）で固定。順を入れ替えると
 * 同じ seed から別の波形が出て、dump した乱数列と torch 参照の対応も崩れる（`host/random.ts`）。
 */
export const synthesizeSbv2 = async (
  state: Sbv2State,
  request: Sbv2GenerateRequest,
): Promise<Sbv2Synthesis> => {
  if (request.text.length === 0) throw new Sbv2InputError("Sbv2Pipeline: text が空");
  const defaults = state.config.defaults;
  const speaker = request.speaker ?? defaults.speaker;
  const style = request.style ?? defaults.style;
  const styleWeight = finiteKnob(request.styleWeight ?? defaults.styleWeight, "styleWeight");
  const knobs: Sbv2Knobs = {
    sdpRatio: finiteKnob(request.sdpRatio ?? defaults.sdpRatio, "sdpRatio"),
    noiseScale: finiteKnob(request.noiseScale ?? defaults.noiseScale, "noiseScale"),
    noiseScaleW: finiteKnob(request.noiseScaleW ?? defaults.noiseScaleW, "noiseScaleW"),
    lengthScale: finiteKnob(request.lengthScale ?? defaults.lengthScale, "lengthScale"),
  };
  if (knobs.lengthScale <= 0) {
    throw new Sbv2InputError(`lengthScale ${knobs.lengthScale} が正でない`);
  }
  const seed = request.seed ?? 0;
  const styleRow = resolveRow(state.config.styles, style, "スタイル");
  const speakerRow = resolveRow(state.config.speakers, speaker, "話者");

  // --- ① テキスト層（GPU 不要・決定的）-------------------------------------
  // 辞書の取得はモジュール doc の NOTE（hub を経由しない唯一の取得）— 初回だけ取りに出る。
  const dictionary = await ensureDictionary(state);
  const overlay = overlayFor(state, dictionary, request.overlay);
  const analysis = analyzeSbv2Text(dictionary, request.text, state.tokenizer, state.rules, {
    ...(overlay === undefined ? {} : { overlay }),
    ...(request.prosody === undefined ? {} : { prosody: request.prosody }),
    ...(request.givenTone === undefined ? {} : { givenTone: request.givenTone }),
  });
  const phonemes = analysis.ids.phoneIds.length;
  const tokens = analysis.inputIds.length;
  // 表の確保も Session も張る前に落とす（{@link assertTokenLimit} の MUST）。上限は T と P で
  // 同じ 1 つだが、P > T なので**両方**見ないと P の超過だけが門を素通りする。
  assertTokenLimit(tokens, state.config.maxTokens);
  assertPhonemeLimit(phonemes, state.config.maxTokens);

  // --- ② text_encoder（DeBERTa・配布形は使う 1 本だけを出す）---------------
  // quant の低精度ノブは渡さない（配布形の text_encoder は i8 の 1 dtype しか無い）。
  const hidden = await withSession(
    state.gpu,
    state.textEncoder,
    {},
    observer(state, "text_encoder"),
    async (run) => {
      // 相対位置の添字表はグラフ入力（焼き込むと Tmax=512 で 2MiB — ADR 0045 波 3）。
      const relPos = buildRelPosTables(
        tokens,
        state.rules.bertRelPos.positionBuckets,
        state.rules.bertRelPos.maxPosition,
      );
      const outputs = await run({
        input_ids: i32(analysis.inputIds, [1, tokens]),
        attention_mask: i32(analysis.inputIds.map(() => 1), [1, tokens]),
        c2p_pos: relPos.c2pPos,
        p2c_pos: relPos.p2cPos,
      });
      const name = bertHiddenOutput(state.textEncoder.graph.outputs, state.rules.bertHiddenFromEnd);
      const tensor = outputs[name];
      if (tensor === undefined) throw new Error(`text_encoder の出力 '${name}' が無い`);
      return { name, data: asF32(tensor, `text_encoder の出力 '${name}'`) };
    },
  );

  // --- ③ BERT 特徴を音素レベルへ tile 展開 ---------------------------------
  const tiled = tileBertToPhoneLevel(hidden.data, tokens, analysis.word2ph);
  assertTiledBert(tiled, phonemes);

  // --- ④ style_vec / g を表から引く ----------------------------------------
  const styleVec = styleVector(
    state.styles.data,
    state.styles.rows,
    state.styles.cols,
    styleRow,
    styleWeight,
  );
  const g = speakerEmbedding(
    state.speakers.data,
    state.speakers.rows,
    state.speakers.cols,
    speakerRow,
  );

  // --- ⑤ front（enc_p + dp + sdp reverse・4 出力）--------------------------
  const random = new Randn(seed);
  const xMask = ones(phonemes);
  const zNoise = random.normals(2 * phonemes, knobs.noiseScaleW);
  const predicted = await withSession(
    state.gpu,
    state.front,
    state.sessionOptions,
    observer(state, "front"),
    async (run) => {
      const outputs = await run({
        x: i32(analysis.ids.phoneIds, [1, phonemes]),
        x_mask: f32(xMask, [1, 1, phonemes]),
        tone: i32(analysis.ids.toneIds, [1, phonemes]),
        language: i32(analysis.ids.languageIds, [1, phonemes]),
        bert: f32(tiled.data, [1, featureWidth(state.front, "bert"), phonemes]),
        style_vec: styleVec,
        g,
        z_noise: f32(zNoise, [1, 2, phonemes]),
      });
      const mP = outputAt(state.front, outputs, 2);
      return {
        logwSdp: asF32(outputAt(state.front, outputs, 0), "front の logw_sdp"),
        logwDp: asF32(outputAt(state.front, outputs, 1), "front の logw_dp"),
        mP: asF32(mP, "front の m_p"),
        logsP: asF32(outputAt(state.front, outputs, 3), "front の logs_p"),
        channels: mP.shape[1],
      };
    },
  );

  // --- ⑥ ホストグルー（継続長 → フレーム展開 → z_p → 相対位置表）----------
  // 総フレーム数の上限は `durationsToFrames` が展開列を確保する前に見る（`host/duration.ts`）。
  const plan = durationsToFrames(
    predicted.logwSdp,
    predicted.logwDp,
    xMask,
    knobs.sdpRatio,
    knobs.lengthScale,
    state.config.maxFrames,
  );
  const channels = predicted.channels;
  const zpNoise = random.normals(channels * plan.totalFrames);
  const zP = buildZp(
    predicted.mP,
    predicted.logsP,
    plan.expandIdx,
    channels,
    zpNoise,
    knobs.noiseScale,
  );
  const tables = buildRelattnTables(plan.totalFrames);

  // --- ⑦ voice（flow + dec 融合）-------------------------------------------
  const audio = await withSession(
    state.gpu,
    state.voice,
    state.sessionOptions,
    observer(state, "voice"),
    async (run) => {
      const outputs = await run({
        z_p: f32(zP, [1, channels, plan.totalFrames]),
        y_mask: f32(ones(plan.totalFrames), [1, 1, plan.totalFrames]),
        g,
        idx_k: tables.idxK,
        valid: tables.valid,
      });
      return asF32(outputAt(state.voice, outputs, 0), "voice の波形出力");
    },
  );

  const expectedSamples = plan.totalFrames * state.rules.hopLength;
  if (audio.length !== expectedSamples) {
    throw new Error(`波形長 ${audio.length} が hopLength×Ty(${expectedSamples}) と違う`);
  }
  // MUST: 非有限値を黙って返さない。WAV 化は NaN を 0 に丸めるので、沈黙誤値がそのまま
  // 「一部が無音の音声」として出てしまう。
  for (const sample of audio) {
    if (!Number.isFinite(sample)) throw new Error("波形に非有限値が含まれる");
  }

  return {
    // MUST: サンプリング周波数は `symbols.json` が正本（pipelineConfig には持たせない —
    // 導出元の二重保持を作らない）。
    sampleRate: state.rules.samplingRate,
    audio,
    trace: {
      analysis,
      speaker,
      style,
      styleWeight,
      knobs,
      seed,
      bertHiddenOutput: hidden.name,
      xMask,
      zNoise,
      wCeil: plan.wCeil,
      zpNoise,
      channels,
      frames: plan.totalFrames,
    },
  };
};

/**
 * SBV2 のテキスト → 音声パイプライン。
 *
 * 構築は {@link Sbv2Pipeline.fromPretrained}（HF から取得）か {@link Sbv2Pipeline.fromAssets}
 * （取得済みバイト列）だけを入口にする — コンストラクタを private にしてあるのは、manifest
 * 検査と資産の突合を迂回した半端な状態を作れないようにするため（ADR 0008）。
 */
export class Sbv2Pipeline {
  readonly #state: Sbv2State;
  /** generate と dispose の直列化鎖（モジュール doc の「1 本ずつ」を公開 API 側で守る）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: Sbv2State) {
    this.#state = state;
  }

  /**
   * HF リポジトリから取得して組む（`loadManifest` → `resolveFiles` → `fetchAssets` →
   * {@link Sbv2Pipeline.fromAssets} の糖衣）。文字列の `ref` は `{ repo }` と読む（= `main`
   * 追従）。省略時は pin 済みの {@link SBV2_DEFAULT_SOURCE}（ADR 0073 決定 2）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef = SBV2_DEFAULT_SOURCE,
    options: Sbv2FromPretrainedOptions = {},
  ): Promise<Sbv2Pipeline> {
    const repoRef: HubRepoRef = typeof ref === "string" ? { repo: ref } : ref;
    const hubOptions = {
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.headers === undefined ? {} : { headers: options.headers }),
      ...(options.onCacheError === undefined ? {} : { onCacheError: options.onCacheError }),
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.caches === undefined ? {} : { caches: options.caches }),
    };
    const loaded = await loadManifest(repoRef, hubOptions);
    const selection = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const files = resolveFiles(loaded.manifest, selection);
    const assets = await fetchAssets(loaded, files, {
      ...hubOptions,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
    });
    return Sbv2Pipeline.fromAssets({ manifest: loaded.manifest, assets }, {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.dictionary === undefined ? {} : { dictionary: options.dictionary }),
      ...(options.overlay === undefined ? {} : { overlay: options.overlay }),
    });
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel` を全てここで済ませ、
   * **Session は 1 本も張らない**（{@link openSbv2State}）。
   */
  static async fromAssets(
    input: Sbv2Assets,
    options: Sbv2PipelineOptions = {},
  ): Promise<Sbv2Pipeline> {
    return new Sbv2Pipeline(await openSbv2State(input, options));
  }

  /**
   * テキストから音声 1 本を生成する。
   *
   * 同じ seed・同じノブなら同じ波形が出る（乱数もホストグルーも決定的 — `host/random.ts`）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（グラフの同時常駐を作らない — モジュール doc）。
   */
  async generate(request: Sbv2GenerateRequest): Promise<GeneratedAudio> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた生成まで
    // 巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("Sbv2Pipeline: dispose 済みでは生成できない");
    }
    return await this.#chain(async () => {
      const { sampleRate, audio } = await synthesizeSbv2(this.#state, request);
      return { sampleRate, data: audio };
    });
  }

  /**
   * テキストを下書き（句 / モーラ構造 + 音素・トーン）まで解析する（**GPU は張らない** —
   * テキスト層だけを回す）。
   *
   * 返る {@link Sbv2ProsodyDraft.prosody} のアクセント核を直して
   * {@link Sbv2GenerateRequest.prosody} へ戻せば、「解析結果を見て直してから合成する」往復に
   * なる。`tones` をそのまま {@link Sbv2GenerateRequest.givenTone} へ渡す低レベル経路も残る。
   *
   * MUST: 生成の直列化鎖には**載せない**。GPU を張らない解析が、進行中の合成（秒オーダー）の
   * 後ろに並ぶ理由が無い — 辞書取得の「1 度きり」は {@link ensureDictionary} が Promise を
   * 持つことで担保している（鎖ではない）。
   */
  async analyzeProsody(
    text: string,
    options: { readonly overlay?: readonly OverlayEntry[] | OverlayDictionary } = {},
  ): Promise<Sbv2ProsodyDraft> {
    // 判定は generate と同じく呼び出し時点で行う（in-flight の解析は GPU を触らないので、
    // dispose と並走しても flush-before-destroy には関わらない）。
    if (this.#disposal !== undefined) {
      throw new Error("Sbv2Pipeline: dispose 済みでは解析できない");
    }
    const state = this.#state;
    const dictionary = await ensureDictionary(state);
    const overlay = overlayFor(state, dictionary, options.overlay);
    const analysis = analyzeSbv2Text(dictionary, text, state.tokenizer, state.rules, {
      ...(overlay === undefined ? {} : { overlay }),
    });
    return { prosody: analysis.prosody, phones: analysis.phones, tones: analysis.tones };
  }

  /**
   * 解放する。**内部で取得した GPU だけ**破棄する（`options.gpu` で渡された GpuContext は
   * 呼び出し側の所有物なので触らない）。
   *
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy）— 破棄も鎖に
   * 載せることで、待ちと破棄の順序を 1 箇所で決める。2 度目以降も同じ完了を返す（先に返すと
   * 呼び出し側が「破棄済み」と見なして次へ進む）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => closeSbv2State(this.#state));
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
