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
 * ホストで組んだ小グラフ 2 本（{@link "./dit-loop.ts"} の `runDitLoopResident`）を同時に張るが、
 * 重みを持たないノード 5 個ぶんなので VRAM の話には効かない。
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
  type GpuContext,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type DistributionSource,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  resolveFiles,
} from "@karume/hub";

import {
  admitIrodori,
  assetJson,
  assetOpener,
  BACKBONE,
  CAPTION_PROJ,
  CODEC_DECODER,
  CODEC_ENCODER,
  DIT,
  DURATION,
  type IrodoriAdmission,
  SPEAKER,
  TEXT_PROJ,
  TOKENIZER,
} from "./admission.ts";
import { type ConditionState, emptyCondition, encodeSpeaker, rightPad } from "./conditioning.ts";
import {
  type DitLoop,
  type LatentStage,
  runDitLoopOnHost,
  runDitLoopResident,
  type UncondVariant,
} from "./dit-loop.ts";
import {
  asF32,
  bool,
  type EmitEvent,
  emitter,
  f32,
  i32,
  outputAt,
  withStageSession,
} from "./stage.ts";
import type { IrodoriPipelineConfig } from "./config.ts";
import { IrodoriTokenizer, parseIrodoriTokenizerAsset } from "./text/tokenizer.ts";
import {
  assertCodecTileFrames,
  type CodecTile,
  decodeTiles,
  DEFAULT_CODEC_TILE_FRAMES,
  planCodecTiles,
} from "./codec.ts";
import { packCaptionIds, packIds } from "./host/pack.ts";
import {
  buildDitMask,
  type IrodoriSegment,
  SEGMENT_ORDER,
  type SegmentLengths,
} from "./host/mask.ts";
import { rowMean } from "./host/pooling.ts";
import { assertAcceptableSeed, Randn } from "./host/random.ts";
import {
  type SampleBounds,
  sequenceLengthFromLogFrames,
  sequenceLengthFromSeconds,
  type SequencePlan,
} from "./host/round.ts";
import { tSchedule } from "./host/sampler.ts";
import { timestepFrequencies } from "./host/t-embed.ts";
import { findFlatteningPoint, trimmedSampleCount } from "./host/trim.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { type FromPretrainedHubOptions, hubLoadOptions } from "../hub/load-options.ts";
import { loadShardComponents, type ModelComponent } from "../hub/components.ts";

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
   *
   * MUST: `onEvent` の中で同じパイプラインの `generate` / `generateLatent` / `dispose` を
   * await してはならない（直列化鎖の自己デッドロック — 中断は throw で行う）。
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
    /** 呼んだときだけ途中潜在を写して返す（{@link "./dit-loop.ts"} の `latentSnapshot`）。 */
    readonly copyLatents: () => IrodoriLatentSnapshot;
  };

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
  /**
   * 構築の中断。{@link IrodoriPipeline.fromAssets} が段の境目（入口 / 各 `openModel` の間 /
   * トークナイザ解釈 / GPU 取得の前後）で検査する。入口を除く各境目では**イベントループへ
   * 1 度譲ってから**検査するので、同期解析の最中に届いた中断も次の境目で効く
   * （`options.gpu` を渡して await が 1 つも無い経路でも同じ）。
   * {@link IrodoriPipeline.fromPretrained} は同じ 1 本を取得層へも渡すので、**DL と組み立ての
   * どちらの最中でも**同じノブで中断できる（DL 完了後だけ中止ボタンが無反応、を作らない）。
   *
   * 中断の例外は `signal.reason` を**そのまま**投げる（包まない — 消費側が
   * `error === controller.signal.reason` で自分の中断を識別できる）。
   */
  readonly signal?: AbortSignal;
};

/**
 * {@link IrodoriPipeline.fromPretrained} が追加で受ける取得層のオプション（hub へ透過する）。
 * `signal` は構築側と共有なので {@link IrodoriPipelineOptions} が持つ。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` が **HTTP 取得元専用**であることを含め、
 * 欄ごとの説明は {@link FromPretrainedHubOptions} に 1 本化してある。
 */
export type IrodoriFromPretrainedOptions = IrodoriPipelineOptions & FromPretrainedHubOptions;

/** 取得済み資産から直接組むときの入力（hub の `fetchAssets` の返り値をそのまま渡す）。 */
export type IrodoriAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/** {@link IrodoriPipeline} の内部状態（公開面には出さない）。 */
export type IrodoriState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: IrodoriPipelineConfig;
  readonly tokenizer: IrodoriTokenizer;
  readonly backbone: ModelComponent;
  readonly textProj: ModelComponent;
  readonly captionProj: ModelComponent;
  readonly speaker: ModelComponent;
  readonly duration: ModelComponent;
  readonly dit: ModelComponent;
  readonly codecEncoder: ModelComponent;
  readonly codecDecoder: ModelComponent;
  /** `dit` の記号次元 S の名前（導出は {@link "./admission.ts"} の `admitIrodori` の 1 度だけ）。 */
  readonly ditSymbol: string;
  /** 低精度ノブ。**`dit` の Session にだけ**渡す（モジュール doc の MUST）。 */
  readonly ditSessionOptions: SessionOptions;
  readonly onRunDiagnostics?: (
    component: IrodoriRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/**
 * admission を通った材料 + 資産から実行状態を組む。
 *
 * MUST: 資産の解析は **GPU を取りに行く前**に落とす。順序がずれると、GPU の無い環境では
 * 別の例外に化けて「何が悪かったのか」が読み手に伝わらない。GPU 取得後に許される検査は
 * GPU の能力（shader-f16）だけ（ADR 0028）。
 * MUST: Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
 */
const buildIrodoriState = async (
  admitted: IrodoriAdmission,
  assets: IrodoriAssets["assets"],
  options: IrodoriPipelineOptions = {},
): Promise<IrodoriState> => {
  const {
    config,
    quant,
    quantName,
    ditSymbol,
    ditSessionOptions,
    backbone,
    textProj,
    captionProj,
    speaker,
    duration,
    dit,
    codecDecoder,
    codecEncoder,
  } = admitted;

  await settleAbort(options.signal);
  const tokenizer = new IrodoriTokenizer(
    parseIrodoriTokenizerAsset(assetJson(assets, TOKENIZER), TOKENIZER),
  );

  await settleAbort(options.signal);

  // MUST: 宣言された feature は device 作成時にしか要求できない（ADR 0028）。共有 GPU を
  // 渡された場合は要求できないので、能力が足りないことを名指しして落とす（共有 GPU は
  // {@link "./admission.ts"} の `admitIrodori` が既に同じ 1 本で見ているが、自前で取った
  // device はここが唯一の門）。
  const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(quant.gpuFeatures));
  const ownsGpu = options.gpu === undefined;
  try {
    // MUST: GPU 取得**後**の中断検査は try の中に置く — 外に出すと、内部で取った device を
    // 誰も解放できないまま抜ける（feature 検査と同じ後始末に乗せる）。ここでもマクロタスクへ
    // 譲る: `acquireGpu` の await 解決はマイクロタスク継続なので、待機中に積まれたクリック
    // 由来の中断タスクはまだ実行されていない。
    await settleAbort(options.signal);
    assertGpuFeaturesGranted(quant.gpuFeatures, gpu, `IrodoriPipeline: quant '${quantName}'`);
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
      ditSymbol,
      ditSessionOptions,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
    };
  } catch (error) {
    // 内部で取った GPU は、構築に失敗したら誰も解放できなくなるのでここで返す。
    if (ownsGpu) gpu.destroy();
    throw error;
  }
};

/** S の決定に要る数を `pipelineConfig` から引く（入口検査と段 ⑤ が同じ 1 本を読む）。 */
const sampleBounds = (config: IrodoriPipelineConfig): SampleBounds => ({
  frameRate: config.frameRate,
  minSeconds: config.minSeconds,
  maxSeconds: config.maxSeconds,
  sampleRate: config.sampleRate,
  hopLength: config.hopLength,
});

/**
 * 要求ノブの受理集合を**生成に入る前**に見る（`generate` / `generateLatent` の入口）。
 *
 * MUST: 受理集合そのものはここに書き写さない — 各ノブの正本（`assertCodecTileFrames` /
 * `assertAcceptableSeed` / `sequenceLengthFromSeconds`）を呼ぶだけにする。条件を写すと、
 * 片方だけ緩んだときに気づけない。
 *
 * なぜ入口なのか: これらの門は本来 latent 生成の**後**（`planCodecTiles` は decode 直前・
 * `Randn` は段 ⑦・秒指定は段 ⑤）にしかなく、`codecTileFrames: 10` や `seed: 1.5` のような
 * 綴り違いが重み 1.26GB のロードと DiT 全 step を消費してから落ちていた。呼び手の綴り違いの
 * 代償を計算時間で払わせない。
 *
 * NOTE（公開面の挙動が変わる 2 点）: ①`codecTileFrames` は codec を回さない
 * {@link IrodoriPipeline.generateLatent} でも検査対象になる（使わない値でも綴り違いなら落ちる）
 * ②`initialNoise` を渡した生成でも `seed` が検査される（従来は `Randn` を通らないので不正値が
 * 黙って無視されていた）。どちらも「効かないノブを黙って受けない」側へ倒した意図的な変更。
 *
 * NOTE: `export` は GPU 無しで受理集合を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008・`latentSnapshot` と同じ流儀）。
 */
export const assertIrodoriRequest = (
  request: IrodoriGenerateRequest,
  config: IrodoriPipelineConfig,
): void => {
  // 既定値も検査する — 呼び手は `codecTileFrames` を渡す義務を負っていないので、既定タイルと
  // `codecHaloFrames` の関係が壊れた配布形は「渡さなければ通る」形にしてはならない。
  assertCodecTileFrames(
    request.codecTileFrames ?? DEFAULT_CODEC_TILE_FRAMES,
    config.codecHaloFrames,
  );
  if (request.seed !== undefined) assertAcceptableSeed(request.seed);
  // 返り値は捨てる（段 ⑤ が改めて同じ関数で S を決める — ここは受理集合を借りるだけ）。
  if (request.durationSeconds !== undefined) {
    sequenceLengthFromSeconds(request.durationSeconds, sampleBounds(config));
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
  const bounds = sampleBounds(config);
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
   * 配布形から取得して組む（`loadManifest` → `resolveFiles` → **各コンポーネントの
   * グラフ shard だけ**を取って `prepareModel` → 残り資産の `fetchAssets` → 構築）。重み shard は
   * Session を組むときに 1 本ずつ流れる（ADR 0070 — `src/hub/components.ts`）。文字列の
   * `ref` は `{ repo }` と読む（= `main` 追従）。**`ref` は必須**（取得元に既定は無い —
   * `src/hub/repo-ref.ts` の MUST）。
   *
   * 手元の配布形は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）。HF の `owner/name` の綴りの門は通らず、network も CacheStorage も
   * 通らない（{@link IrodoriFromPretrainedOptions} の HTTP 専用ノブは効かない）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: IrodoriFromPretrainedOptions = {},
  ): Promise<IrodoriPipeline> {
    const source = toManifestSource(
      ref,
      "IrodoriPipeline.fromPretrained",
      'IRODORI_SOURCES["irodori-v4.1-small"]（@karume/models/irodori）',
    );
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const selection = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const files = resolveFiles(loaded.manifest, selection);
    // signal は取得層と構築の**両方**へ渡す（DL が終わった瞬間に中断が効かなくなる窓を作らない）。
    const buildOptions: IrodoriPipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...selection,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    };
    // 家族の門は admission 席で通す（重み shard を取る前 — `hub/components.ts`）。
    const { admitted, assets } = await loadShardComponents(
      "IrodoriPipeline.fromPretrained",
      loaded,
      files,
      [BACKBONE, TEXT_PROJ, CAPTION_PROJ, SPEAKER, DURATION, DIT, CODEC_DECODER, CODEC_ENCODER],
      async (open) => {
        const admitted = await admitIrodori(loaded.manifest, open, buildOptions);
        // 配布形が宣言した `requiredLimits` は**重み shard を取る前**にここで見る
        // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `IrodoriPipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      },
    );
    return new IrodoriPipeline(await buildIrodoriState(admitted, assets, buildOptions));
  }

  /**
   * 取得済みの manifest + 資産から組む。契約検査・資産の解釈・`openModel` を全てここで済ませ、
   * **Session は 1 本も張らない**。{@link IrodoriPipelineOptions.signal} を渡すと段の境目で
   * 中断できる（`admitIrodori` の NOTE）。
   *
   * 取得キーは `resolveFiles` の規約どおり **2 形とも受ける** — 素の 1 本（`dit`）と、
   * shard 分割形（`dit[0]` / `dit[1]` / …）。分割形は
   * バイト列を連結せず `fromPretrained` と同じ shard 逐次面へ流す。添字の欠番と素キーとの混在は
   * fail loudly（受け口の実装は `src/hub/components.ts` の 1 本）。
   */
  static async fromAssets(
    input: IrodoriAssets,
    options: IrodoriPipelineOptions = {},
  ): Promise<IrodoriPipeline> {
    const admitted = await admitIrodori(input.manifest, assetOpener(input.assets), options);
    return new IrodoriPipeline(await buildIrodoriState(admitted, input.assets, options));
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
    // 要求ノブの検査も呼び出し時点（鎖の外）で行う — 鎖に入れると、先行生成の決着まで
    // 待たされてから綴り違いで落ちる（`createOperationChain` は空の鎖でも 1 microtask 遅れる）。
    assertIrodoriRequest(request, this.#state.config);
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
    assertIrodoriRequest(request, this.#state.config);
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
