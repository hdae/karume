/**
 * `IrodoriPipeline` — テキスト（+ caption + 参照話者）→ **波形** の 1 本の面。
 *
 * 出口は 2 つある: {@link IrodoriPipeline.generate}（波形）と
 * {@link IrodoriPipeline.generateLatent}（patch 済み latent `[S,32]` — codec を回さない）。
 * 後者は latent 門（`e2e_irodori_latent_test.ts`）の基盤であり、埋め込みだけが要る使い方の口。
 *
 * 参照話者は 3 通りで渡せる（`IrodoriSpeakerInput`）— 参照音声そのもの（`audio`・下の段 0）/
 * その DACVAE latent（`latent`）/ 出来合いの speaker state（`stateOverride`）。
 *
 * パイプライン（NN は全段 Karume・torch 不使用）:
 *
 * 0. ホスト: 参照音声を 120 秒で切り詰め → LUFS −16 正規化 → reflect pad →
 *    `codec_encoder` で latent へ（`audio` を渡したときだけ — `host/reference.ts`）
 * 1. ホスト: `normalize_text` → BOS 前置 → **詰めた** token 列（静的方式 — pad で呼ばない）
 * 2. `backbone`（ModernBERT）を text / caption の 2 回（**同じ 1 セッション**）
 * 3. `text-proj` → text 条件 / `caption-proj`（2 出力）→ caption 条件 + `caption_vec`
 * 4. ホスト: 参照 latent を patch → `speaker` → **平均トークンを前置**（グラフの外）
 * 5. `duration` → log frames → ホストで S を決める（expm1 → 銀行家丸め → clamp）
 * 6. ホスト: 条件 state を Tmax へ右 pad + 区間マスクを組む
 * 7. `dit` を 1 セッションで 40〜100 forward（CFG 合成と Euler は GPU 常駐の小グラフ 2 本 —
 *    ループ全体が 1 batch で、ホストへ降りるのは最後の潜在 1 回だけ）→ latent
 * 8. ホスト: 末尾トリムの位置を **z 上で**決める（`host/trim.ts`）
 * 9. `codec_decoder` を 1 セッションでタイルぶん回す（`codec.ts`）→ 全長の波形
 * 10. ホスト: 秒指定 / 末尾トリムの短いほうでサンプル単位に切る
 *
 * ## MUST: グラフは段ごとに開いて閉じる（`dit` と `codec_decoder` だけが複数 run）
 *
 * {@link IrodoriPipeline.fromAssets} は **Session を 1 本も張らない** — 開くのはコンテナ
 * （`openModel` = ヘッダ解析のみ）までで、GPU 常駐は {@link IrodoriPipeline.generate} の
 * 中で段ごとに張っては畳む。`backbone` だけで 1.26GB あるので、条件エンコーダと DiT を
 * 同時に生かさない。codec も同じ理由で DiT を畳んでから張る。DiT の段だけは `dit` に加えて
 * ホストで組んだ小グラフ 2 本（{@link runDitLoopResident}）を同時に張るが、重みを持たない
 * ノード 5 個ぶんなので VRAM の話には効かない。
 *
 * MUST: この段取りは**公開 API 側でも**守る — `generate` / `generateLatent` は直列化鎖に載せ
 * （並行呼び出しは待たされて順に走る）、`dispose` はその完了を待ってから GPU を破棄する。
 * 載せないと、並行呼び出し 2 本ぶんのグラフが同時常駐し、生成中の dispose が
 * flush-before-destroy を破る。
 *
 * ## NOTE: `codec_encoder` はタイル分割しない（decoder と非対称）
 *
 * 参照音声は 1 回の encode で流す。encoder の中間テンソルは 120 秒（T = 3000）で 1.47GB × 2 に
 * 達するので、**`maxStorageBufferBindingSize` が既定の 128MiB しか無い機では長い参照で確保に
 * 失敗する**（decoder 側の {@link DEFAULT_CODEC_TILE_FRAMES} に相当するものが無い）。参照音声は
 * 数秒〜十数秒が普通で、そこでは単発が通る。タイル化が要ると分かったら decoder と同じ形
 * （halo 付きの平行移動同変）をもう 1 本入れる。
 *
 * ## MUST: 低精度ノブが効くのは `dit` だけ
 *
 * quant の `session` は `dit` の Session にだけ渡す。条件エンコーダ 5 本は 1 回ずつしか
 * 回らず（合成時間の支配項は 40〜100 forward の DiT）、実行形ノブの比較軸を DiT に保つ。
 *
 * ## MUST: uncond は「cond の state + 該当区間のマスク全 False」だけ（ADR 0047 決定 1）
 *
 * `speakerUncondMode` / `cfgGuidanceMode` が対応外の配布形は **`config.ts` が parse 時に
 * 拒否する**ので、ここに分岐は無い。
 *
 * ## MUST: 数値の正はここでは担保されない
 *
 * 正はグラフ単位の golden E2E（`packages/runtime/tests/e2e_irodori_test.ts`）と、full-loop の
 * latent golden（`tools/exporter/irodori_pipeline.py` が出す `pipeline/case.*.safetensors`）が
 * 担保する。
 */

import {
  acquireGpu,
  createSession,
  type GpuContext,
  type KarumeModel,
  openModel,
  type ResidentTensor,
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

import {
  IRODORI_PIPELINE_MAJOR,
  IRODORI_PIPELINE_NAME,
  type IrodoriPipelineConfig,
  parseIrodoriPipelineConfig,
} from "./config.ts";
import { IrodoriTokenizer, parseIrodoriTokenizerAsset } from "./text/tokenizer.ts";
import { type CodecTile, decodeTiles, DEFAULT_CODEC_TILE_FRAMES, planCodecTiles } from "./codec.ts";
import { packCaptionIds, packIds } from "./host/pack.ts";
import {
  buildDitMask,
  type IrodoriSegment,
  SEGMENT_ORDER,
  type SegmentLengths,
} from "./host/mask.ts";
import { patchReferenceLatent } from "./host/patch.ts";
import { prependMeanToken, rowMean } from "./host/pooling.ts";
import { Randn } from "./host/random.ts";
import { normalizeReference, reflectPadToHop } from "./host/reference.ts";
import {
  type SampleBounds,
  sequenceLengthFromLogFrames,
  sequenceLengthFromSeconds,
  type SequencePlan,
} from "./host/round.ts";
import { type CfgVariant, combineCfg, eulerStep, tSchedule } from "./host/sampler.ts";
import {
  COMBINE_INPUTS,
  COMBINE_OUTPUT,
  combineGraph,
  EULER_INPUTS,
  EULER_OUTPUT,
  eulerGraph,
} from "./host/sampler-graph.ts";
import { timestepEmbedding, timestepFrequencies } from "./host/t-embed.ts";
import { findFlatteningPoint, trimmedSampleCount } from "./host/trim.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { toSessionOptions } from "../session/options.ts";

/** manifest の weights / assets 表に現れる取得キー（ADR 0041 §3 の規約名）。 */
const BACKBONE = "backbone";
const TEXT_PROJ = "text_proj";
const CAPTION_PROJ = "caption_proj";
const SPEAKER = "speaker";
const DURATION = "duration";
const DIT = "dit";
const CODEC_DECODER = "codec_decoder";
const CODEC_ENCODER = "codec_encoder";
const TOKENIZER = "tokenizer";

/** 生成結果。`data` は patch 済み latent `[frames × latentDim]`（行優先）。 */
export type GeneratedLatent = {
  readonly data: Float32Array<ArrayBuffer>;
  /** latent のフレーム数 S。 */
  readonly frames: number;
  readonly latentDim: number;
  /**
   * 実際に使った乱数 seed。`initialNoise` を渡した生成では乱数を 1 度も引かないので `undefined`。
   */
  readonly seed?: number;
  /** `dit` を回した回数（cond + CFG の uncond）。 */
  readonly forwards: number;
};

/**
 * 生成結果（波形）。`data` は**切り出し済み**のモノラル f32（`encodeWav` へそのまま渡せる）。
 *
 * 名前に接頭辞が付いているのは、barrel（`mod.ts`）が SBV2 の `GeneratedAudio` を既に出して
 * いるため（あちらは `{ sampleRate, data }` の 2 欄で、こちらは latent 側の観測値も返す）。
 */
export type IrodoriGeneratedAudio = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
  /** 生成した latent のフレーム数 S。**波形長は末尾トリムでこれより短くなりうる**。 */
  readonly frames: number;
  /** 実際に使った乱数 seed。`initialNoise` を渡した生成では `undefined`。 */
  readonly seed?: number;
  /** `dit` を回した回数（cond + CFG の uncond）。 */
  readonly forwards: number;
};

/**
 * 参照話者の与え方。
 *
 * - `audio` — 参照音声そのもの（モノラル f32 + サンプリング周波数）。`decodeWav` の返り値を
 *   そのまま渡せる。ホストで正規化 → `codec_encoder` を通してから `latent` と同じ経路へ合流
 *   する。**周波数は配布形の `sampleRate` と一致していなければならない**（リサンプルは
 *   持たない — 違えば fail loudly）。
 * - `latent` — 参照音声の DACVAE latent `[frames × latentDim]`。patch → `speaker` →
 *   平均トークン前置の**正規経路**を通る。
 * - `stateOverride` — 既に作ってある speaker state `[rows × speakerDim]`。`speaker` グラフも
 *   `speaker_norm` も平均トークン前置も**通さず**そのまま条件として使う（上流の
 *   `speaker_state_override` — 埋め込みを配る運用のための口）。
 */
export type IrodoriSpeakerInput =
  | { readonly audio: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number } }
  | { readonly latent: Float32Array<ArrayBuffer> }
  | { readonly stateOverride: Float32Array<ArrayBuffer> };

/** 1 回の生成要求。 */
export type IrodoriGenerateRequest = {
  readonly text: string;
  /** 声色の指示文（Voice Design）。空 / 未指定なら caption 条件はゼロ供給 + CFG off。 */
  readonly caption?: string;
  readonly speaker?: IrodoriSpeakerInput;
  /** 乱数 seed（既定 0 — 同じ seed なら同じ latent）。 */
  readonly seed?: number;
  /**
   * 初期ノイズ `[frames × latentDim]` の直接注入。**再現・検証用の口**で、通常の生成では
   * 使わない（統合門が torch の乱数列をそのまま食わせるために置いてある）。長さは決まった
   * S に一致していなければならないので、`durationSeconds` と併せて使うのが普通。
   */
  readonly initialNoise?: Float32Array<ArrayBuffer>;
  /**
   * 発話長の直接指定（秒）。渡すと `duration` グラフを**回さない**（上流の `manual_seconds`）。
   */
  readonly durationSeconds?: number;
  /**
   * codec decode を 1 回あたり何 latent フレームに割るか（既定
   * {@link DEFAULT_CODEC_TILE_FRAMES}）。**性能とメモリのノブで、出力は変わらない** —
   * halo を捨てた採用区間は全長 decode とビット一致する（`codec.ts` のモジュール doc）。
   * S がこの値以下なら 1 枚に縮退する（= 単発 decode）。
   */
  readonly codecTileFrames?: number;
  /**
   * 生成イベントの観測席（{@link IrodoriGenerateEvent}）— 段の開始 / 終了と DiT の 1 step
   * 完了ごとに呼ばれる。
   *
   * **await する**（発火の順序が決定的になり、消費側で間引き / スロットリングができる）。
   * **例外は握らない**（`onRunDiagnostics` と同じ流儀 = fail loudly）— 副産物として
   * **throw が step 粒度の中断手段**になる（生成は reject し、Session は `withSession` の
   * `finally` で解放される）。
   *
   * MUST: 指定すると DiT ループは**ホスト経路**で回る（`gpuTiming` と同じ選択機構）。常駐経路
   * （ADR 0054）は全 step を 1 batch + 単一フェンスに束ねる設計なので途中の観測が構造的に
   * 不可能で、`enqueue` 時点の発火は「進捗」として嘘になる。**代償は壁時計だけ** — ADR 0054 が
   * DiT ループに与えた 1.76 倍を放棄する（生成全体では実測 7.2 → 8.6 秒 / S 170・参照環境
   * 2026-08-16）。2 経路の出力はビット同一（`runDitLoopResident` の MUST）なので**波形は
   * 1 ビットも変わらない**（`e2e_irodori_wav_test.ts` の onEvent 段が同じ sha256 で門にしている）。
   */
  readonly onEvent?: (event: IrodoriGenerateEvent) => void | Promise<void>;
};

/**
 * {@link IrodoriPipelineOptions.onRunDiagnostics} が受けるコンポーネント名。
 * `stage` イベント（{@link IrodoriGenerateEvent}）の段名も同じ 8 名。
 */
export type IrodoriRunComponent =
  | "backbone"
  | "text-proj"
  | "caption-proj"
  | "speaker"
  | "duration"
  | "dit"
  | "codec-encoder"
  | "codec-decoder";

/** `denoise-step` の `copyLatents()` が返す途中潜在の写し。 */
export type IrodoriLatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  /** 潜在の形 `[frames, latentDim]`。 */
  readonly shape: readonly number[];
};

/**
 * {@link IrodoriGenerateRequest.onEvent} が受ける生成イベント。
 *
 * NOTE: anima の `AnimaGenerateEvent` と同型だが `vae-tile` に当たる席は無い — codec decode の
 * タイルは**性能とメモリのノブ**（`codecTileFrames`）で、段としては `codec-decoder` の
 * `stage` が覆う。
 */
export type IrodoriGenerateEvent =
  /** 段の Session 構築の**前**（`start`）と解放の**後**（`end`）— GB 級ロードの進捗が見える。 */
  | {
    readonly kind: "stage";
    readonly component: IrodoriRunComponent;
    readonly at: "start" | "end";
  }
  | {
    readonly kind: "denoise-step";
    /** 完了した step 数（1-based）。**CFG の内側の forward 数ではない**。 */
    readonly step: number;
    readonly steps: number;
    /** その step で消費した時刻 `t`（flow matching のスケジュール — anima の sigma に当たる）。 */
    readonly t: number;
    /** 呼んだときだけ途中潜在を写して返す（{@link latentSnapshot}）。 */
    readonly copyLatents: () => IrodoriLatentSnapshot;
  };

/** 生成イベントの発火口（未購読なら何もしない 1 本に畳んで、発火点に分岐を置かない）。 */
type EmitEvent = (event: IrodoriGenerateEvent) => Promise<void>;

/** 未購読のときの発火口。 */
const NO_EVENTS: EmitEvent = () => Promise.resolve();

/** 要求の `onEvent` を発火口に畳む（await して例外は握らない — {@link IrodoriGenerateRequest.onEvent}）。 */
const emitter = (
  onEvent: ((event: IrodoriGenerateEvent) => void | Promise<void>) | undefined,
): EmitEvent =>
  onEvent === undefined ? NO_EVENTS : async (event: IrodoriGenerateEvent) => {
    await onEvent(event);
  };

/**
 * 途中潜在を返す口を作る（**lazy copy** — 呼ばれたときだけ写す）。
 *
 * 進捗だけを購読する消費側にコピー費用が一切かからず、内部の配列を渡さないので「次 step の
 * 入力を購読側に握られる」事故も構造的に起きない。
 *
 * MUST: 呼ばれた時点ではなく**作った時点**の配列を写す（引数で束縛する）。DiT ループの `x` は
 * step ごとに**新しい配列へ差し替わる**ので、この束縛がそのまま「その step の潜在」になる。
 * ループ変数を閉じ込めると、後から呼んだ購読側に別 step の潜在が返る。
 *
 * NOTE: `export` は GPU 無しで独立性を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const latentSnapshot = (
  latent: Float32Array<ArrayBuffer>,
  shape: readonly number[],
): () => IrodoriLatentSnapshot =>
(): IrodoriLatentSnapshot => ({ data: new Float32Array(latent), shape });

/** 構築オプション（{@link IrodoriPipeline.fromAssets} / {@link IrodoriPipeline.fromPretrained} 共通）。 */
export type IrodoriPipelineOptions = {
  /**
   * 既存の GPU を共有する。**渡した側が所有権を持つ**ので {@link IrodoriPipeline.dispose} は
   * 破棄しない。省略時はパイプラインが内部で `acquireGpu` し、`dispose()` で破棄する。
   */
  readonly gpu?: GpuContext;
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * 実行 1 回ごとの診断を受け取る観測席（1 生成 = 条件エンコーダ 5〜7 回 + `dit` 40〜100 回）。
   * op 別 GPU 時間（`lastRunTiming`）が要るときは `gpu` に
   * `acquireGpu({ gpuTiming: true })` を渡す（ADR 0021 — 既定は計測しない）。
   *
   * NOTE: 計測を有効にした device では DiT ループが**ホスト経路**（forward ごとに readback）で
   * 回る — 常駐経路が使う batch は計測と両立しない（`beginBatch` が拒否する）。出力は同じだが
   * 壁時計は倍近くになるので、内訳を採るとき以外は計測を有効にしない。DiT の診断は
   * `enqueue` ごとに届き、常駐経路では `lastRun` / `lastRunTiming` が `undefined` になる
   * （アリーナも計測窓も作らない）。
   *
   * コールバックの例外は握らない（fail loudly — 生成ごと落ちる）。
   */
  readonly onRunDiagnostics?: (
    component: IrodoriRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/** {@link IrodoriPipeline.fromPretrained} だけが使う取得層のオプション（hub へ透過する）。 */
export type IrodoriFromPretrainedOptions = IrodoriPipelineOptions & {
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
export type IrodoriAssets = {
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
      `irodori: 資産 '${key}' が無い（manifest の weights / assets に ${key} が要る）` +
        `（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
    );
  }
  const bytes = assets[key];
  if (bytes.byteOffset !== 0 || bytes.byteLength !== bytes.buffer.byteLength) {
    throw new Error(
      `irodori: 資産 '${key}' の bytes が buffer 全体を占めていない` +
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
 * コンテナ 8 本が要る）。`mod.ts` / サブパス面には出さない（ADR 0008）。
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
    throw new Error(`irodori: 資産 '${key}' が UTF-8 として読めない`, { cause });
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new Error(`irodori: 資産 '${key}' が JSON として読めない`, { cause });
  }
};

/**
 * グラフ入力の 1 軸ぶんの**静的**次元が `pipelineConfig` の宣言と一致することを見る。
 *
 * MUST: 落とさない。条件 state の宣言長や幅がホストの値とずれても、右 pad と行数計算は
 * そのまま通り（shape は合う）、**別の位置の条件を読んだ**結果が沈黙で出る。
 */
const assertStaticDim = (
  model: KarumeModel,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => {
  const spec = model.graph.inputs.find((input) => input.name === inputName);
  if (spec === undefined) throw new Error(`irodori: グラフ入力 '${inputName}' が無い（${where}）`);
  const dim = spec.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `irodori: ${where} — グラフ入力 '${inputName}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

/**
 * グラフ**出力**の 1 軸が「記号 × 係数」の派生次元で、その係数が宣言と一致することを見る。
 *
 * MUST: 落とさない。decoder の出力倍率（1 latent フレーム → `hopLength` サンプル）がずれても
 * 出力は「それらしい長さの波形」になり、秒指定の切り出しと末尾トリムだけが別の位置を指す。
 */
const assertOutputScale = (
  model: KarumeModel,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`irodori: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const symbol = model.graph.symbols[0];
  // 正準表記は `coeff·sym` で、**係数 1 は省略**する（`format/dims.ts` の `formatDim`）。
  // 綴りを合わせないと、倍率 1 の正しいグラフをここが誤って拒否する。
  const canonical = `${expected === 1 ? "" : expected}${symbol}`;
  const dim = value.shape[axis];
  if (symbol === undefined || dim !== canonical) {
    throw new Error(
      `irodori: ${where} — グラフ出力 '${name}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig からの期待は '${canonical}'`,
    );
  }
};

/**
 * グラフ**出力**の 1 軸が宣言どおりの**静的**次元であることを見る。
 *
 * MUST: 落とさない。encoder の latent 幅が `pipelineConfig` の `latentDim` とずれても、後段の
 * patch は「幅 × patchSize」で割り切れる限り通ってしまい、**別のチャネルを話者特徴として
 * 読んだ**結果が沈黙で出る。
 */
const assertOutputDim = (
  model: KarumeModel,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`irodori: グラフ出力 '${name}' の宣言が無い（${where}）`);
  }
  const dim = value.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `irodori: ${where} — グラフ出力 '${name}' の軸 ${axis} が ${String(dim)}、` +
        `pipelineConfig は ${expected}`,
    );
  }
};

const asF32 = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

const f32 = (data: Float32Array<ArrayBuffer>, shape: readonly number[]): Tensor => ({
  dtype: "f32",
  shape: [...shape],
  data,
});

const i32 = (data: Int32Array<ArrayBuffer>, shape: readonly number[]): Tensor => ({
  dtype: "i32",
  shape: [...shape],
  data,
});

/** bool の実表現は u32 の 0 / 1（ADR 0009）。 */
const bool = (value: boolean): Tensor => ({
  dtype: "bool",
  shape: [1, 1],
  data: Uint32Array.of(value ? 1 : 0),
});

/** グラフ出力**名**を位置で引く（IR v1 の出力名は `output.<i>` — 名前を決め打ちしない）。 */
const outputNameAt = (model: KarumeModel, index: number): string => {
  const name = model.graph.outputs[index];
  if (name === undefined) {
    throw new Error(`グラフ出力 ${index} が無い（${model.graph.outputs.length} 本しかない）`);
  }
  return name;
};

/** グラフ出力を**位置**で引く。 */
const outputAt = (
  model: KarumeModel,
  outputs: Readonly<Record<string, Tensor>>,
  index: number,
): Tensor => {
  const name = outputNameAt(model, index);
  const tensor = outputs[name];
  if (tensor === undefined) throw new Error(`グラフ出力 ${index}（'${name}'）が実行結果に無い`);
  return tensor;
};

/**
 * 1 グラフぶんの Session を張り、使い終わったら必ず解放する。
 * MUST: `finally` で dispose する — 途中で落ちたときに VRAM が残ると、後続の段が確保に
 * 失敗して「最初の失敗とは別の場所」で落ちる。
 *
 * NOTE: `sbv2/pipeline.ts` / `anima/pipeline.ts` にも同名の helper がある（**意図的な重複**）。
 * 3 ファミリで `run` の返し方が違い（1 出力 / 多出力 / ここは同一セッションでの複数 run）、
 * 共通化すると全呼び出し側の分解が変わって実 GPU でしか露見しない回帰リスクを負う。
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

/** 観測席（{@link IrodoriPipelineOptions.onRunDiagnostics}）へコンポーネント名を焼いて渡す。 */
const observer = (
  state: IrodoriState,
  component: IrodoriRunComponent,
): ((diagnostics: SessionDiagnostics) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  return listener === undefined ? undefined : (diagnostics) => listener(component, diagnostics);
};

/**
 * 段 1 本を回す（`stage` イベントを Session 構築の前と解放の後に挟む）。
 * 途中で落ちたら `end` は出ない（生成ごと reject する — `onEvent` の doc）。
 */
const withStageSession = async <T>(
  state: IrodoriState,
  emit: EmitEvent,
  component: IrodoriRunComponent,
  model: KarumeModel,
  sessionOptions: SessionOptions,
  body: (
    run: (inputs: Record<string, Tensor>) => Promise<Record<string, Tensor>>,
    session: Session,
  ) => Promise<T>,
): Promise<T> => {
  await emit({ kind: "stage", component, at: "start" });
  const result = await withSession(
    state.gpu,
    model,
    sessionOptions,
    observer(state, component),
    body,
  );
  await emit({ kind: "stage", component, at: "end" });
  return result;
};

/** {@link IrodoriPipeline} の内部状態（公開面には出さない）。 */
type IrodoriState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: IrodoriPipelineConfig;
  readonly tokenizer: IrodoriTokenizer;
  readonly backbone: KarumeModel;
  readonly textProj: KarumeModel;
  readonly captionProj: KarumeModel;
  readonly speaker: KarumeModel;
  readonly duration: KarumeModel;
  readonly dit: KarumeModel;
  readonly codecEncoder: KarumeModel;
  readonly codecDecoder: KarumeModel;
  /** 低精度ノブ。**`dit` の Session にだけ**渡す（モジュール doc の MUST）。 */
  readonly ditSessionOptions: SessionOptions;
  readonly onRunDiagnostics?: (
    component: IrodoriRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/** 条件 1 本ぶんの中間状態（行数を値と一緒に持ち歩く — 幅は config が正本）。 */
type ConditionState = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly rows: number;
};

/** 条件を載せない（参照なし / caption 空）ときのゼロ供給。右 pad で全 0 行になる。 */
const emptyCondition = (): ConditionState => ({ data: new Float32Array(0), rows: 0 });

/** 条件 state を宣言長へ右詰め 0 pad する（ADR 0047 のホスト残置）。 */
const rightPad = (
  state: ConditionState,
  rows: number,
  width: number,
  where: string,
): Float32Array<ArrayBuffer> => {
  if (state.rows > rows) {
    throw new Error(
      `IrodoriPipeline: ${where} の長さ ${state.rows} が宣言長 ${rows} を超えている`,
    );
  }
  if (state.data.length !== state.rows * width) {
    throw new Error(
      `IrodoriPipeline: ${where} の要素数 ${state.data.length} が ${state.rows}×${width} と違う`,
    );
  }
  const padded = new Float32Array(rows * width);
  padded.set(state.data);
  return padded;
};

/**
 * manifest + 資産から実行状態を組む（{@link IrodoriPipeline.fromAssets} の中身）。
 *
 * MUST: manifest の契約違反と**資産の解析・グラフとの突合**は **GPU を取りに行く前**に落とす。
 * 順序がずれると、GPU の無い環境では別の例外に化けて「何が悪かったのか」が読み手に
 * 伝わらない。GPU 取得後に許される検査は GPU の能力（shader-f16）だけ（ADR 0028）。
 * MUST: Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
 */
const openIrodoriState = async (
  input: IrodoriAssets,
  options: IrodoriPipelineOptions = {},
): Promise<IrodoriState> => {
  const { manifest, assets } = input;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `IrodoriPipeline: model '${modelName}' は manifest に無い` +
        `（利用可能: ${manifest.available.models.join(" / ")}）`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== IRODORI_PIPELINE_NAME) {
    throw new Error(
      `IrodoriPipeline: manifest の pipeline が '${name}/${major}'` +
        `（'${IRODORI_PIPELINE_NAME}/${IRODORI_PIPELINE_MAJOR}' が必要）`,
    );
  }
  if (major !== IRODORI_PIPELINE_MAJOR) {
    // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
    throw new Error(
      `IrodoriPipeline: pipeline '${name}/${major}' の major に未対応` +
        `（この実装が読めるのは ${IRODORI_PIPELINE_NAME}/${IRODORI_PIPELINE_MAJOR}）`,
    );
  }
  const config = parseIrodoriPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `IrodoriPipeline: quant '${quantName}' は manifest に無い` +
        `（利用可能: ${entry.available.quants.join(" / ")}）`,
    );
  }
  const quant = entry.quants[quantName];
  const wantsShaderF16 = quant.gpuFeatures?.shaderF16 === true;

  const backbone = openModel(assetBuffer(assets, BACKBONE));
  const textProj = openModel(assetBuffer(assets, TEXT_PROJ));
  const captionProj = openModel(assetBuffer(assets, CAPTION_PROJ));
  const speaker = openModel(assetBuffer(assets, SPEAKER));
  const duration = openModel(assetBuffer(assets, DURATION));
  const dit = openModel(assetBuffer(assets, DIT));
  const codecDecoder = openModel(assetBuffer(assets, CODEC_DECODER));
  const codecEncoder = openModel(assetBuffer(assets, CODEC_ENCODER));
  const tokenizer = new IrodoriTokenizer(
    parseIrodoriTokenizerAsset(assetJson(assets, TOKENIZER), TOKENIZER),
  );

  // グラフの宣言と pipelineConfig の突合（ホストの式が読む数は全て config 由来）。
  assertStaticDim(dit, "x_t", 2, config.latentDim, "latentDim");
  assertStaticDim(dit, "t_embed", 1, config.timestepEmbedDim, "timestepEmbedDim");
  assertStaticDim(dit, "text_state", 1, config.maxTextLen, "maxTextLen");
  assertStaticDim(dit, "text_state", 2, config.textDim, "textDim");
  assertStaticDim(dit, "speaker_state", 1, config.speakerRows, "speakerRows");
  assertStaticDim(dit, "speaker_state", 2, config.speakerDim, "speakerDim");
  assertStaticDim(dit, "caption_state", 1, config.maxCaptionLen, "maxCaptionLen");
  assertStaticDim(dit, "caption_state", 2, config.captionDim, "captionDim");
  assertStaticDim(duration, "text_state", 2, config.textDim, "textDim");
  assertStaticDim(duration, "speaker_vec", 1, config.speakerDim, "speakerDim");
  assertStaticDim(duration, "caption_vec", 1, config.captionDim, "captionDim");
  // 参照 latent の patch 幅（latentDim × speakerPatchSize）が speaker の入力幅と一致する。
  assertStaticDim(
    speaker,
    "latent",
    2,
    config.latentDim * config.speakerPatchSize,
    "latentDim × speakerPatchSize",
  );
  // codec decoder は latent を 1 フレーム = hopLength サンプルへ展開する。入力幅と**出力の
  // 派生次元の係数**の両方を見る（係数だけがずれた資産は shape が合ったまま通り、切り出しと
  // 末尾トリムのサンプル位置だけが静かに別の場所を指す）。
  assertStaticDim(codecDecoder, "latent", 2, config.latentDim, "latentDim");
  assertOutputScale(codecDecoder, 2, config.hopLength, "hopLength");
  // encoder は逆向き（`[1,T,hopLength]` の波形 → `[1,T,latentDim]`）。入力のフレーム幅が
  // `hopLength` でないと、ホストが並べた波形が**1 フレームずつずれて**読まれる。
  assertStaticDim(codecEncoder, "wav", 2, config.hopLength, "hopLength");
  assertOutputDim(codecEncoder, 2, config.latentDim, "latentDim");

  const ditSessionOptions = toSessionOptions(quant.session);

  // MUST: `shader-f16` は device 作成時にしか要求できない（ADR 0028）。共有 GPU を渡された
  // 場合は要求できないので、能力が足りないことを**ここで**名指しして落とす。
  const gpu = options.gpu ?? await acquireGpu(wantsShaderF16 ? { shaderF16: true } : {});
  const ownsGpu = options.gpu === undefined;
  if (wantsShaderF16 && !gpu.shaderF16Enabled) {
    // 内部で取った GPU は、ここで投げると誰も解放できなくなるので返してから落とす。
    if (ownsGpu) gpu.destroy();
    throw new Error(
      `IrodoriPipeline: quant '${quantName}' は shader-f16 を要求するが、渡された` +
        " GpuContext で有効になっていない（acquireGpu({ shaderF16: true }) で取り直す）",
    );
  }
  return {
    gpu,
    ownsGpu,
    config,
    tokenizer,
    backbone,
    textProj,
    captionProj,
    speaker,
    duration,
    dit,
    codecEncoder,
    codecDecoder,
    ditSessionOptions,
    ...(options.onRunDiagnostics === undefined
      ? {}
      : { onRunDiagnostics: options.onRunDiagnostics }),
  };
};

/**
 * 参照音声 → DACVAE latent（ホスト前処理 + `codec_encoder`）。
 *
 * 切り詰めの上限は `speakerRows` から導く — speaker 条件は「平均トークン 1 本 + patch した
 * 参照」なので、載る参照は `(speakerRows − 1) × speakerPatchSize` フレーム（実重み v4-small で
 * 3,000 フレーム = 120 秒）。**TS 側に秒数を定数で置かない**（config.ts の MUST — 重みを
 * 差し替えたときにホストだけ古い上限を持つ形を作らない）。
 *
 * MUST: 切り詰めは正規化より**前**。後ろに回すと、捨てる区間の音量が LUFS に混ざる
 * （上流も `wav[:, :int(max_ref_seconds·sr)]` を先に取る）。
 */
const encodeReferenceAudio = async (
  state: IrodoriState,
  emit: EmitEvent,
  audio: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number },
): Promise<Float32Array<ArrayBuffer>> => {
  const { config } = state;
  if (audio.sampleRate !== config.sampleRate) {
    // リサンプルは持たない（ADR 0048 の流儀 — 黙って近似せず、変換は呼び出し側の責務にする）。
    throw new Error(
      `IrodoriPipeline: 参照音声が ${audio.sampleRate}Hz（配布形は ${config.sampleRate}Hz）` +
        " — リサンプルは持たないので、あらかじめ変換して渡す",
    );
  }
  const maxSamples = (config.speakerRows - 1) * config.speakerPatchSize * config.hopLength;
  const limited = audio.data.length > maxSamples
    ? (audio.data.slice(0, maxSamples) as Float32Array<ArrayBuffer>)
    : audio.data;
  const padded = reflectPadToHop(
    normalizeReference(limited, config.sampleRate).data,
    config.hopLength,
  );
  const frames = padded.length / config.hopLength;
  return await withStageSession(
    state,
    emit,
    "codec-encoder",
    state.codecEncoder,
    {},
    async (run) => {
      const outputs = await run({ wav: f32(padded, [1, frames, config.hopLength]) });
      return asF32(outputAt(state.codecEncoder, outputs, 0), "codec encoder の出力");
    },
  );
};

/** speaker 条件を組む（参照音声 / 参照 latent / 埋め込み直接指定 / 参照なしのゼロ短絡）。 */
const encodeSpeaker = async (
  state: IrodoriState,
  emit: EmitEvent,
  input: IrodoriSpeakerInput | undefined,
): Promise<ConditionState> => {
  const { config } = state;
  if (input === undefined) {
    // 参照なしはグラフを回さずゼロを置く（上流の `no_ref` と厳密に一致することは exporter の
    // `_no_reference_evidence` が実測済み）。区間マスクも全 0 になるので寄与は厳密に 0。
    return emptyCondition();
  }
  if ("stateOverride" in input) {
    const { stateOverride } = input;
    if (stateOverride.length === 0 || stateOverride.length % config.speakerDim !== 0) {
      throw new Error(
        `IrodoriPipeline: speaker.stateOverride の長さ ${stateOverride.length} が` +
          ` speakerDim ${config.speakerDim} の正の倍数でない`,
      );
    }
    // MUST: `speaker` グラフも `speaker_norm` も平均トークン前置も通さない（上流
    // `encode_conditions` の `speaker_state_override` 経路）。加工すると、配られた埋め込みが
    // 二重に正規化された別のベクトルとして条件に入る。
    return { data: stateOverride, rows: stateOverride.length / config.speakerDim };
  }
  const latent = "audio" in input
    ? await encodeReferenceAudio(state, emit, input.audio)
    : input.latent;
  const patched = patchReferenceLatent(latent, config.latentDim, config.speakerPatchSize);
  const encoded = await withStageSession(
    state,
    emit,
    "speaker",
    state.speaker,
    {},
    async (run) => {
      const outputs = await run({
        latent: f32(patched.data, [1, patched.tokens, patched.width]),
      });
      return asF32(outputAt(state.speaker, outputs, 0), "speaker の出力");
    },
  );
  // 平均トークンの前置はグラフの外（ADR 0047 決定 4）。
  return {
    data: prependMeanToken(encoded, patched.tokens, config.speakerDim),
    rows: patched.tokens + 1,
  };
};

/**
 * latent 段の結果。`plan.targetSamples` は波形の切り出しに要るので**決めた場所から持ち回る**
 * （`GeneratedLatent` は公開の面なので混ぜない）。
 */
type LatentStage = {
  readonly latent: GeneratedLatent;
  readonly plan: SequencePlan;
};

/** CFG の 1 変種（落とす区間・強さ・その区間だけ False にしたマスク）。 */
type UncondVariant = {
  readonly segment: IrodoriSegment;
  readonly scale: number;
  readonly mask: Tensor;
};

/**
 * DiT ループ 1 本ぶんの材料（2 つの経路が**同じもの**を読む — ホストの計算はどちらでも同一）。
 *
 * 条件 3 本を値と Tensor の両方で持つのは、常駐経路が {@link ResidentTensor.write} に生の
 * 配列を要り、ホスト経路が `run` に Tensor を要るため。**同じ配列の別の見方**であって、
 * 独立に更新される複製ではない。
 */
type DitLoop = {
  readonly frames: number;
  /** 初期ノイズ `[frames × latentDim]`。 */
  readonly initial: Float32Array<ArrayBuffer>;
  readonly schedule: Float32Array<ArrayBuffer>;
  readonly frequencies: Float32Array<ArrayBuffer>;
  /** 右 pad 済みの条件 3 本（グラフ入力名 → 値）。 */
  readonly conditionValues: Readonly<Record<string, Float32Array<ArrayBuffer>>>;
  readonly conditions: Readonly<Record<string, Tensor>>;
  readonly condMask: Tensor;
  readonly uncondVariants: readonly UncondVariant[];
};

/** ループの結果（最終潜在と `dit` を回した回数）。 */
type DitLoopResult = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly forwards: number;
};

/**
 * forward ごとにホストへ降りるループ（`run` → readback → `combineCfg` + `eulerStep` →
 * 再アップロード）。
 *
 * **数値の正本**であり、計測が有効な device（`gpuTiming` — 常駐経路が使う batch を開けない）
 * と生成イベントの購読（`onEvent` — 1 batch の途中は観測できない）での唯一の経路でもある。
 */
const runDitLoopOnHost = async (
  state: IrodoriState,
  emit: EmitEvent,
  loop: DitLoop,
): Promise<DitLoopResult> => {
  const { config } = state;
  let x = loop.initial;
  let forwards = 0;
  await withStageSession(
    state,
    emit,
    "dit",
    state.dit,
    state.ditSessionOptions,
    async (run) => {
      for (let step = 0; step < config.steps; step += 1) {
        const t = loop.schedule[step];
        const tNext = loop.schedule[step + 1];
        const tEmbed = f32(timestepEmbedding(t, loop.frequencies), [1, config.timestepEmbedDim]);
        const xTensor = f32(x, [1, loop.frames, config.latentDim]);
        const cond = asF32(
          outputAt(
            state.dit,
            await run({ x_t: xTensor, t_embed: tEmbed, mask: loop.condMask, ...loop.conditions }),
            0,
          ),
          "dit の速度場",
        );
        forwards += 1;
        const variants: CfgVariant[] = [];
        if (t >= config.cfgMinT && t <= config.cfgMaxT) {
          // MUST: 合成順は SEGMENT_ORDER（text → speaker → caption）— `combineCfg` の doc。
          for (const variant of loop.uncondVariants) {
            const outputs = await run({
              x_t: xTensor,
              t_embed: tEmbed,
              mask: variant.mask,
              ...loop.conditions,
            });
            forwards += 1;
            variants.push({
              scale: variant.scale,
              velocity: asF32(
                outputAt(state.dit, outputs, 0),
                `dit の速度場（uncond ${variant.segment}）`,
              ),
            });
          }
        }
        x = eulerStep(x, combineCfg(cond, variants), Math.fround(tNext - t));
        await emit({
          kind: "denoise-step",
          step: step + 1,
          steps: config.steps,
          t,
          copyLatents: latentSnapshot(x, [loop.frames, config.latentDim]),
        });
      }
    },
  );
  return { x, forwards };
};

/**
 * ループ全体を **1 batch** に束ねる GPU 常駐経路（H-5）。
 *
 * 潜在・速度場・CFG の途中結果・条件 3 本を全て {@link ResidentTensor} に置き、`dit` と
 * ホストで組んだ小グラフ 2 本（{@link combineGraph} / {@link eulerGraph}）を `enqueue` で
 * 積むだけにする。ホストへ降りるのは最後の 1 回（`x_t.read()`）だけで、フェンスは
 * `batch.finish()` の 1 本に集約される。
 *
 * MUST: 区間の中で `Session.run` を待たない（自己デッドロック — `beginBatch` の doc）。
 * MUST: 演算の積み方は {@link runDitLoopOnHost} と 1 演算ずつ同型（変種順・差の基準・
 * 引数順）。ずれると最終桁が動き、WAV sha256 門が割れる。強さ `scale` はこちらが GPU へ
 * 渡す前に f32 へ丸めるのに対しホスト経路は JS の f64 で乗算するが、`parseCfgScales` が
 * f32 厳密な値しか受理しないので、2 経路の出力一致は**配布形に依らず無条件で**成立する。
 * MUST: 常駐テンソルを返すのは Session を全て畳んだ**後**（焼き込み参照が残っていると
 * `dispose` が fail loudly になる）。
 */
const runDitLoopResident = async (state: IrodoriState, loop: DitLoop): Promise<DitLoopResult> => {
  const { config, gpu } = state;
  const { frames } = loop;
  const observe = observer(state, "dit");
  const latentBytes = frames * config.latentDim * 4;
  // 記号次元は常駐入力から束縛できない（常駐テンソルは shape を持たない）ので毎 enqueue 明示する。
  const symbols = state.dit.graph.symbols;
  if (symbols.length !== 1) {
    throw new Error(`irodori: dit の記号次元が 1 本でない（[${symbols.join(", ")}]）`);
  }
  const bindings = { [symbols[0]]: frames };
  const velocity = outputNameAt(state.dit, 0);
  const residents: ResidentTensor[] = [];
  const sessions: Session[] = [];
  try {
    const createResident = async (bytes: number, label: string): Promise<ResidentTensor> => {
      const tensor = await gpu.createResident(bytes, label);
      residents.push(tensor);
      return tensor;
    };
    const xT = await createResident(latentBytes, "irodori.x_t");
    const vCond = await createResident(latentBytes, "irodori.v_cond");
    const vVariant = await createResident(latentBytes, "irodori.v_variant");
    const accumulator = await createResident(latentBytes, "irodori.cfg_acc");
    xT.write(loop.initial);
    const conditions: Record<string, ResidentTensor> = {};
    for (const [name, values] of Object.entries(loop.conditionValues)) {
      const tensor = await createResident(values.byteLength, `irodori.${name}`);
      // ループの前に 1 度だけ投入する（毎 forward の 3.8MB writeBuffer が丸ごと消える）。
      tensor.write(values);
      conditions[name] = tensor;
    }
    const open = async (model: KarumeModel, options: SessionOptions): Promise<Session> => {
      const session = await createSession(gpu, model, options);
      sessions.push(session);
      return session;
    };
    const dit = await open(state.dit, state.ditSessionOptions);
    const combine = await open(openModel(combineGraph(frames, config.latentDim)), {});
    const euler = await open(openModel(eulerGraph(frames, config.latentDim)), {});
    // 強さは step に依らないので 1 度だけ作る。
    const scales = loop.uncondVariants.map((variant) => f32(Float32Array.of(variant.scale), [1]));

    let forwards = 0;
    const batch = await gpu.beginBatch();
    try {
      for (let step = 0; step < config.steps; step += 1) {
        const t = loop.schedule[step];
        const tEmbed = f32(timestepEmbedding(t, loop.frequencies), [1, config.timestepEmbedDim]);
        const forward = async (mask: Tensor, target: ResidentTensor): Promise<void> => {
          await dit.enqueue(
            { x_t: xT, t_embed: tEmbed, mask, ...conditions },
            { batch, bindings, copyOutputs: { [velocity]: target } },
          );
          forwards += 1;
          if (observe !== undefined) observe(dit.diagnostics());
        };
        await forward(loop.condMask, vCond);
        const guided = t >= config.cfgMinT && t <= config.cfgMaxT &&
          loop.uncondVariants.length > 0;
        if (guided) {
          // MUST: 合成順は SEGMENT_ORDER（text → speaker → caption）— `combineCfg` の doc。
          for (let index = 0; index < loop.uncondVariants.length; index += 1) {
            await forward(loop.uncondVariants[index].mask, vVariant);
            // k = 0 の被加数は cond そのもの（正本 `combineCfg` の `let value = base`）。同じ
            // バッファを acc_in と cond の 2 口で読むだけなので WebGPU 上も合法。
            await combine.enqueue({
              [COMBINE_INPUTS.accumulator]: index === 0 ? vCond : accumulator,
              [COMBINE_INPUTS.cond]: vCond,
              [COMBINE_INPUTS.variant]: vVariant,
              [COMBINE_INPUTS.scale]: scales[index],
            }, { batch, copyOutputs: { [COMBINE_OUTPUT]: accumulator } });
          }
        }
        await euler.enqueue({
          [EULER_INPUTS.x]: xT,
          [EULER_INPUTS.velocity]: guided ? accumulator : vCond,
          [EULER_INPUTS.deltaT]: f32(
            Float32Array.of(Math.fround(loop.schedule[step + 1] - t)),
            [1],
          ),
        }, { batch, copyOutputs: { [EULER_OUTPUT]: xT } });
      }
    } finally {
      // MUST: 区間は必ず閉じる（開いたままだと device 単位のロックが返らず、以後の run が
      // 永久に待つ）。
      await batch.finish();
    }
    return { x: new Float32Array(await xT.read()), forwards };
  } finally {
    for (const session of sessions) await session.dispose();
    for (const tensor of residents) tensor.dispose();
  }
};

/** テキスト 1 本（+ caption / 参照話者）から latent を作る。 */
const generateLatent = async (
  state: IrodoriState,
  emit: EmitEvent,
  request: IrodoriGenerateRequest,
): Promise<LatentStage> => {
  const { config } = state;
  const textIds = packIds(state.tokenizer, request.text, config.maxTextLen, "text");
  // caption の有無は**生の文字列**で決める（上流 `str(req.caption).strip() != ""`）。
  const captionText = request.caption ?? "";
  const captionIds = captionText.trim().length === 0
    ? undefined
    : packCaptionIds(state.tokenizer, captionText, config.maxCaptionLen);

  // --- ① backbone（text / caption を同じ 1 セッションで回す）---------------
  // 1.26GB の重みを 2 度アップロードしないため 1 セッション 2 run にしてある。診断は run
  // ごとに "backbone" として届く。
  const hidden = await withStageSession(
    state,
    emit,
    "backbone",
    state.backbone,
    {},
    async (run) => {
      const text = outputAt(
        state.backbone,
        await run({ input_ids: i32(textIds, [1, textIds.length]) }),
        0,
      );
      if (captionIds === undefined) return { text, caption: undefined };
      const captionHidden = outputAt(
        state.backbone,
        await run({ input_ids: i32(captionIds, [1, captionIds.length]) }),
        0,
      );
      return { text, caption: captionHidden };
    },
  );

  // --- ② text-proj -------------------------------------------------------
  const textState = await withStageSession(
    state,
    emit,
    "text-proj",
    state.textProj,
    {},
    async (run) => {
      const output = outputAt(state.textProj, await run({ hidden: hidden.text }), 0);
      return { data: asF32(output, "text-proj の出力"), rows: output.shape[1] };
    },
  );

  // --- ③ caption-proj（2 出力 — 第 2 出力は `caption_norm` 済み系列）------
  let captionState: ConditionState = emptyCondition();
  let captionVec: Float32Array<ArrayBuffer> | undefined;
  if (hidden.caption !== undefined) {
    const captionHidden = hidden.caption;
    const encoded = await withStageSession(
      state,
      emit,
      "caption-proj",
      state.captionProj,
      {},
      async (run) => {
        const outputs = await run({ hidden: captionHidden });
        const raw = outputAt(state.captionProj, outputs, 0);
        const normed = outputAt(state.captionProj, outputs, 1);
        return {
          state: { data: asF32(raw, "caption-proj の第 1 出力"), rows: raw.shape[1] },
          normed: asF32(normed, "caption-proj の第 2 出力"),
        };
      },
    );
    captionState = encoded.state;
    // `caption_vec` は **`caption_norm` 済み**系列の行平均（第 1 出力から採ると別のベクトル）。
    captionVec = rowMean(encoded.normed, captionState.rows, config.captionDim);
  }

  // --- ④ speaker ---------------------------------------------------------
  const speakerState = await encodeSpeaker(state, emit, request.speaker);
  const hasSpeaker = speakerState.rows > 0;

  // --- ⑤ S の決定 --------------------------------------------------------
  const bounds: SampleBounds = {
    frameRate: config.frameRate,
    minSeconds: config.minSeconds,
    maxSeconds: config.maxSeconds,
    sampleRate: config.sampleRate,
    hopLength: config.hopLength,
  };
  let plan: SequencePlan;
  if (request.durationSeconds !== undefined) {
    // 手動指定は duration グラフを回さない（上流 `manual_seconds` 経路）。
    plan = sequenceLengthFromSeconds(request.durationSeconds, bounds);
  } else {
    const logFrames = await withStageSession(
      state,
      emit,
      "duration",
      state.duration,
      {},
      async (run) => {
        const outputs = await run({
          text_state: f32(textState.data, [1, textState.rows, config.textDim]),
          // `speaker_vec` は speaker state の**先頭行**（= 平均トークン）。
          speaker_vec: f32(
            hasSpeaker
              ? speakerState.data.slice(0, config.speakerDim)
              : new Float32Array(config.speakerDim),
            [1, config.speakerDim],
          ),
          has_speaker: bool(hasSpeaker),
          caption_vec: f32(captionVec ?? new Float32Array(config.captionDim), [
            1,
            config.captionDim,
          ]),
          has_caption: bool(captionVec !== undefined),
        });
        return asF32(outputAt(state.duration, outputs, 0), "duration の出力")[0];
      },
    );
    plan = sequenceLengthFromLogFrames(logFrames, bounds);
  }
  const { frames } = plan;
  if (frames > config.ditSymMax) {
    throw new Error(
      `IrodoriPipeline: 決まった latent 長 ${frames} が dit の宣言上限 ${config.ditSymMax} を超えている`,
    );
  }

  // --- ⑥ 条件の右 pad と区間マスク（ADR 0047 決定 1 / 4）------------------
  const caps: SegmentLengths = {
    text: config.maxTextLen,
    speaker: config.speakerRows,
    caption: config.maxCaptionLen,
  };
  const used: SegmentLengths = {
    text: textState.rows,
    speaker: speakerState.rows,
    caption: captionState.rows,
  };
  const conditionValues = {
    text_state: rightPad(textState, caps.text, config.textDim, "text 条件"),
    speaker_state: rightPad(speakerState, caps.speaker, config.speakerDim, "speaker 条件"),
    caption_state: rightPad(captionState, caps.caption, config.captionDim, "caption 条件"),
  };
  const conditions = {
    text_state: f32(conditionValues.text_state, [1, caps.text, config.textDim]),
    speaker_state: f32(conditionValues.speaker_state, [1, caps.speaker, config.speakerDim]),
    caption_state: f32(conditionValues.caption_state, [1, caps.caption, config.captionDim]),
  };
  const maskShape = [1, 1, 1, frames + caps.text + caps.speaker + caps.caption];
  const condMask: Tensor = {
    dtype: "bool",
    shape: maskShape,
    data: buildDitMask(frames, used, caps),
  };
  // CFG が有効な条件（上流 `has_*_cfg`）: 強さが正で、かつその条件が実際に載っていること
  // （text は必ず 1 token 以上ある — 空文字は packIds が落とす）。uncond マスクは step ごとに
  // 作り直さず、ここで 1 度だけ組む。
  const uncondVariants: readonly UncondVariant[] = SEGMENT_ORDER
    .filter((segment: IrodoriSegment) => config.cfgScales[segment] > 0 && used[segment] > 0)
    .map((segment) => ({
      segment,
      // MUST: 強さは f32 で持つ — ホスト経路の `combineCfg` は f64 のまま乗算に入れ、常駐経路は
      // GPU へ渡す前に f32 へ丸めるので、値が f32 非厳密だと 2 経路が 1〜2 ulp 割れる。
      // `parseCfgScales` が f32 厳密しか受理しないので現物では恒等だが、ここで丸めておくと
      // 「ホスト側が見るのは f32」が局所で読める（数値の正本は `host/sampler.ts`）。
      scale: Math.fround(config.cfgScales[segment]),
      mask: {
        dtype: "bool",
        shape: maskShape,
        data: buildDitMask(frames, used, caps, segment),
      } satisfies Tensor,
    }));

  // --- ⑦ Euler + CFG independent -----------------------------------------
  const noiseLength = frames * config.latentDim;
  const seed = request.seed ?? 0;
  let initial: Float32Array<ArrayBuffer>;
  if (request.initialNoise === undefined) {
    initial = new Randn(seed).normals(noiseLength);
  } else {
    if (request.initialNoise.length !== noiseLength) {
      throw new Error(
        `IrodoriPipeline: initialNoise の長さ ${request.initialNoise.length} が` +
          ` ${frames}×${config.latentDim} と違う（決まった latent 長は ${frames}）`,
      );
    }
    initial = request.initialNoise;
  }
  const loop: DitLoop = {
    frames,
    initial,
    schedule: tSchedule(config.steps, config.initScale),
    frequencies: timestepFrequencies(config.timestepEmbedDim),
    conditionValues,
    conditions,
    condMask,
    uncondVariants,
  };
  // MUST: 計測が有効な device では batch を開けない（ADR 0021 — 未回収の timestamp が区間ぶん
  // 溜まる）。op 別 GPU 時間の観測席（`onRunDiagnostics` + `gpuTiming`）を残すため、その device
  // だけは従来のホストループへ落とす。積む演算はどちらも同型で、出力は同じでなければならない。
  // 生成イベントの購読（`onEvent`）も同じ理由でホスト経路を選ぶ — 1 batch + 単一フェンスの
  // 常駐経路は step の完了そのものがホストから観測できない（`onEvent` の doc の MUST）。
  const { x, forwards } = state.gpu.gpuTimingEnabled || request.onEvent !== undefined
    ? await runDitLoopOnHost(state, emit, loop)
    : await runDitLoopResident(state, loop);

  return {
    latent: {
      data: x,
      frames,
      latentDim: config.latentDim,
      ...(request.initialNoise === undefined ? { seed } : {}),
      forwards,
    },
    plan,
  };
};

/**
 * latent を波形へ落とす（タイルを 1 セッションで順に回す）。
 *
 * MUST: **全長ぶん**の波形を組む。末尾トリムと秒指定の切り出しは decode の**後**に波形の
 * サンプル単位で行う（latent を切ってから decode すると境界 padding が変わり、全長 decode の
 * 先頭部分とビット一致しない — `codec.ts` のモジュール doc）。
 */
const decodeWaveform = async (
  state: IrodoriState,
  emit: EmitEvent,
  latent: GeneratedLatent,
  tiles: readonly CodecTile[],
): Promise<Float32Array<ArrayBuffer>> => {
  const { latentDim, hopLength } = state.config;
  return await withStageSession(
    state,
    emit,
    "codec-decoder",
    state.codecDecoder,
    {},
    async (run) =>
      await decodeTiles(latent.data, { latentDim, hopLength, tiles }, async (slice, frames) => {
        const outputs = await run({ latent: f32(slice, [1, frames, latentDim]) });
        return asF32(outputAt(state.codecDecoder, outputs, 0), "codec decoder の出力");
      }),
  );
};

/** テキスト 1 本から波形を作る（latent → 末尾トリム → decode → 切り出し）。 */
const generateAudio = async (
  state: IrodoriState,
  emit: EmitEvent,
  request: IrodoriGenerateRequest,
): Promise<IrodoriGeneratedAudio> => {
  const { config } = state;
  const { latent, plan } = await generateLatent(state, emit, request);
  // 末尾トリムの判定は z 上（decode 前）— 上流 `_synthesize` と同じ順序。
  const flattening = findFlatteningPoint(latent.data, latent.frames, config.latentDim);
  const samples = trimmedSampleCount(plan.targetSamples, flattening, config.hopLength);
  const tiles = planCodecTiles(latent.frames, {
    tileFrames: request.codecTileFrames ?? DEFAULT_CODEC_TILE_FRAMES,
    haloFrames: config.codecHaloFrames,
  });
  const waveform = await decodeWaveform(state, emit, latent, tiles);
  const data = samples === waveform.length
    ? waveform
    : (waveform.slice(0, samples) as Float32Array<ArrayBuffer>);
  // MUST: 非有限値を黙って返さない。WAV 化は NaN を 0 に丸め ±Inf をフルスケールへ張り付かせる
  // ので、沈黙誤値が「一部だけ無音 / 一部だけ轟音の音声」として出てしまう。
  for (const sample of data) {
    if (!Number.isFinite(sample)) throw new Error("irodori: 波形に非有限値が含まれる");
  }
  return {
    data,
    sampleRate: config.sampleRate,
    frames: latent.frames,
    ...(latent.seed === undefined ? {} : { seed: latent.seed }),
    forwards: latent.forwards,
  };
};

/**
 * Irodori-TTS v4 のテキスト → 音声パイプライン。
 *
 * 構築は {@link IrodoriPipeline.fromPretrained}（HF から取得）か
 * {@link IrodoriPipeline.fromAssets}（取得済みバイト列）だけを入口にする — コンストラクタを
 * private にしてあるのは、manifest 検査と資産の突合を迂回した半端な状態を作れないようにする
 * ため（ADR 0008）。
 */
export class IrodoriPipeline {
  readonly #state: IrodoriState;
  /** generate / generateLatent と dispose の直列化鎖（「1 本ずつ」を公開 API 側で守る）。 */
  readonly #chain = createOperationChain();
  /**
   * dispose の 1 本。**undefined でないことが「dispose 済み」**（別に真偽値を持つと、独立に
   * 更新される派生状態になる）。
   */
  #disposal: Promise<void> | undefined;

  private constructor(state: IrodoriState) {
    this.#state = state;
  }

  /**
   * HF リポジトリから取得して組む（`loadManifest` → `resolveFiles` → `fetchAssets` →
   * {@link IrodoriPipeline.fromAssets} の糖衣）。文字列の `ref` は `{ repo }` と読む。
   */
  static async fromPretrained(
    ref: string | HubRepoRef,
    options: IrodoriFromPretrainedOptions = {},
  ): Promise<IrodoriPipeline> {
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
    return IrodoriPipeline.fromAssets({ manifest: loaded.manifest, assets }, {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    });
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel` を全てここで済ませ、
   * **Session は 1 本も張らない**。
   */
  static async fromAssets(
    input: IrodoriAssets,
    options: IrodoriPipelineOptions = {},
  ): Promise<IrodoriPipeline> {
    return new IrodoriPipeline(await openIrodoriState(input, options));
  }

  /**
   * テキストから波形 1 本を生成する（`encodeWav` へそのまま渡せる f32 モノラル）。
   *
   * 同じ seed・同じ要求なら同じ波形が出る（乱数もホストグルーも決定的 — `host/random.ts`）。
   *
   * 並行に呼ばれた場合は**待たされて順に**走る（グラフの同時常駐を作らない — モジュール doc）。
   */
  async generate(request: IrodoriGenerateRequest): Promise<IrodoriGeneratedAudio> {
    // dispose 済みの判定は呼び出し時点で行う（鎖の中で見ると、dispose より前に受けた生成まで
    // 巻き添えで落ちる）。
    if (this.#disposal !== undefined) {
      throw new Error("IrodoriPipeline: dispose 済みでは生成できない");
    }
    return await this.#chain(() => generateAudio(this.#state, emitter(request.onEvent), request));
  }

  /**
   * テキストから latent 1 本を生成する（codec を回さない — latent 門と埋め込み用途の面）。
   *
   * 同じ seed・同じ要求なら同じ latent が出る。{@link IrodoriPipeline.generate} と**同じ鎖**に
   * 載るので、混ぜて並行に呼んでも順に走る。
   */
  async generateLatent(request: IrodoriGenerateRequest): Promise<GeneratedLatent> {
    if (this.#disposal !== undefined) {
      throw new Error("IrodoriPipeline: dispose 済みでは生成できない");
    }
    return await this.#chain(async () =>
      (await generateLatent(this.#state, emitter(request.onEvent), request)).latent
    );
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
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` 対応（Explicit Resource Management）— {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
