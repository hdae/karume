/**
 * `WanPipeline` — テキスト → 動画（Wan2.1 T2V 1.3B）の 1 本の面（ADR 0118 決定 7）。
 *
 * パイプライン（全段 Karume・torch 不使用）:
 *
 * 1. **text** — 経路は構築時に選ぶ（{@link WanPipelineOptions.textEncoder} — ADR 0119 決定 7・追記
 *    「段 10d の設計」B）:
 *    - `"gpu"`（既定）— umT5（重み i8 per-channel・活性 f32 — ADR 0119 決定 5）の Session を張り、
 *      positive と negative を 1 回ずつ回して畳む。プロンプトは `prompt_clean` の鏡像とトークナイザの門を
 *      通る任意の文字列（2〜512 トークン — ADR 0119 決定 1・2・4）。出力 `[1, L, 4096]` をホストが
 *      DiT の文脈の行数までゼロで詰める
 *    - `"precomputed"` — テキスト埋め込み資産からプロンプトの行を引くだけ（ADR 0118 決定 4 の第 1 段・
 *      GPU を使わない）。資産の集合に無い文字列は fail loudly
 * 2. **transformer** — S 形の DiT（`[1,S,64]` — 決定 3）を steps 回。CFG は uncond → cond の逐次 2 回
 *    （B = 1）で、合成 `uncond + g·(cond − uncond)` と UniPC の更新はホスト（決定 5 — `scheduler.ts`）
 * 3. **vae_decoder** — chunk グラフ 2 本（first / next）の**常時タイル** decode（決定 2 —
 *    `vae-tiles.ts`）→ `[-1, 1]` へクランプ
 *
 * 返すのはフレーム `[3, F, H, W]` の f32（値域 `[-1, 1]`）。PNG への書き出しは呼び手
 * （`wanFrameToRgba` + `encodePng`）。
 *
 * ## MUST: 段ごとに Session を張って畳む・text → DiT → VAE の順に 1 段ずつ
 *
 * 構築（{@link WanPipeline.fromPretrained} / {@link WanPipeline.fromAssets}）では Session を 1 本も
 * 張らない（コンテナを開くまで）。
 * `generate` の中で段ごとに Session を張り、畳んでから次の段を張る（決定 7 — anima の既定
 * `"per-stage"` と同じ）。text 段（umT5 の重み i8 5.30 GiB）と DiT 段（81 フレームで 7.31 GiB）の和は
 * B570 の天井 9,600 MiB を越える（ADR 0119 決定 11）ので、text 段は DiT 段を張る前に畳む。DiT 段
 * （33 フレームで約 5.7 GiB）と VAE 段（約 2.9 GiB）も同時に持たない。text 段を畳んだ直後の確保の残り
 * （DiT の後には初回だけ見えた）は未測（ADR 0119 未解決）。
 *
 * NOTE（解放待ちは入れていない — 決定 7 の「切り替えのピークを測り、足りなければ入れる」の結果）:
 * B570（Intel / wgpu）は `destroy()` の解放が次の device poll まで遅れうる（docs/known-issues.md
 * 「Intel Arc B570」節）が、2026-10-02 の実測（832×480・33 フレーム・2 ステップの通し・fdinfo の
 * `drm-total-vram0`）では、DiT 段の山 5.75 GiB が **DiT の Session を畳んだ直後**（`stage` の `end`・
 * VAE の段を張る前）に 0.43 GiB（最初の段の前 0.19 GiB）へ戻っていた。切り替えの山は DiT 段の山
 * そのもの（VAE 段の山 2.88 GiB と重ならない）。遅れが出ないのは、`Session.dispose` が未完了の仕事を
 * 決着させてから破棄し、破棄の時点で GPU が参照中のバッファが残らないため（推測 — 遅れの実測例は
 * 参照中のバッファを捨てる形だった）。
 *
 * MUST: 段の切り替えは公開 API 側でも守る — `generate` は直列化鎖に載せ（並行呼び出しは待たされて
 * 順に走る）、`dispose` はその完了を待ってから GPU を破棄する。
 *
 * ## 中断（`signal` — ADR 0119 決定 9・追記「段 10d の設計」G）
 *
 * 構築は {@link WanPipelineOptions.signal}（取得層と構築の段の境目 — anima / irodori と同じ形）、生成は
 * {@link WanGenerateRequest.signal}（段の境目・各 run の前〈text の 2 回・DiT の各 step〉・VAE のタイルの
 * 間 — `generation/sequence.ts` の形）。検査はイベントループへ 1 度譲ってから見る（`settleAbort`）。
 * 1 回の run は不可分で、中断は次の境目で効く。中断の例外は開いている Session を畳んでから
 * `signal.reason` を包まずに投げる（消費側が `error === controller.signal.reason` で識別できる —
 * ADR 0083 決定 5）。
 *
 * ## 配布形（`karume-wan2.1` — ADR 0118 段 7）
 *
 * 入口は配布形から取得する {@link WanPipeline.fromPretrained} と、取得済みのバイト列（manifest +
 * 資産）から組む {@link WanPipeline.fromAssets} の 2 つで、どちらも同じ家族 admission
 * （{@link WanPipeline.#admit}）と組み立て（{@link WanPipeline.#build}）を通る。生成の既定（step 数・
 * guidance・shift）は manifest の `pipelineConfig`（`config.ts`）が持ち、UniPC の構造は上流の値
 * （`WAN_UNIPC_CONFIG`）のまま。
 *
 * NOTE: 公開配布リポの対応表（`WAN_SOURCES`）はまだ無い — 公開リポを持たない家族は表を持たない
 * （ADR 0073 決定 1・vowel-detector と同じ）。HF 公開の pin 焼き込みの回に足す（docs/release-runbook.md）。
 *
 * ## MUST: 出力の「正しさ」はここでは担保されない
 *
 * 数値の正は参照との e2e（実 GPU — `packages/models/tests/e2e_wan_pipeline_test.ts`）が担保する。
 * seed 付き乱数は torch の `randn` とは別列（`random.ts`）なので、参照との照合は初期ノイズを
 * {@link WanGenerateRequest.latents} で注入して行う。
 */

import {
  acquireGpu,
  codecLayout,
  type GpuContext,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  type DistributionSource,
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  type Manifest,
  type ModelEntry,
  type Quant,
  resolveSelection,
} from "@karume/hub";

import { ModelInputError } from "../errors.ts";
import { assertAcceptableSeed } from "../request-gates.ts";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import { readAssetBuffer, readAssetJson, readWholeAsset } from "../hub/asset-readers.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadContainerComponents,
  type ModelComponent,
} from "../hub/components.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import {
  parseWanPipelineConfig,
  WAN_PIPELINE_MAJOR,
  WAN_PIPELINE_NAME,
  type WanPipelineConfig,
} from "./config.ts";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
  wanTokenWidth,
} from "./dit-tokens.ts";
import { parseWanRopeBase, type WanRopeBase, wanRopeTables, wanRopeWidth } from "./dit-rope.ts";
import { timestepsProj } from "./dit-timestep.ts";
import { denormalizeWanLatents } from "./latents.ts";
import { WanRandn } from "./random.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  type WanUniPcSchedule,
  wanUniPcSchedule,
} from "./scheduler.ts";
import {
  findWanTextEmbedding,
  padWanTextEmbedding,
  parseWanTextEmbeds,
  type WanPrompt,
  type WanTextEmbedding,
  type WanTextEmbeds,
} from "./text-embeds.ts";
import { parseWanTokenizerAsset, WanPromptEncoder } from "./text/tokenizer.ts";
import { buildUmt5RelativePositionBuckets, WAN_UMT5_MAX_LENGTH } from "./umt5/relative-position.ts";
import {
  padUmt5Context,
  umt5SessionInputs,
  WAN_UMT5_INPUT_IDS,
  WAN_UMT5_RELATIVE_POSITION_BUCKETS,
} from "./umt5/session-io.ts";
import { WanVaeChunkCaches, type WanVaeChunkLayout, wanVaeChunkLayout } from "./vae-chunks.ts";
import {
  clampWanVaeFrames,
  decodeWanVaeTiled,
  planWanVaeTiles,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "./vae-tiles.ts";

/** 部品のキー（系列のグラフ名 = 段 7 の manifest の weights 名）。 */
const TRANSFORMER = "transformer";
const VAE_DECODER_FIRST = "vae_decoder_first";
const VAE_DECODER_NEXT = "vae_decoder_next";
/**
 * umT5 の部品（ADR 0119 追記「段 10d の設計」D — manifest では umT5 の配布リポへの越境参照）。取るのは
 * `"gpu"` の経路だけ。
 */
const TEXT_ENCODER = "text_encoder";

/** テキストエンコーダの経路（{@link WanPipelineOptions.textEncoder}）。 */
type WanTextEncoderRoute = NonNullable<WanPipelineOptions["textEncoder"]>;

/** 既定の経路（ADR 0119 決定 7 — 同じ文字列で経路が黙って変わらないよう、自動の切り替えは持たない）。 */
const DEFAULT_TEXT_ENCODER: WanTextEncoderRoute = "gpu";

/**
 * 経路ごとに取る部品（取得面の `componentKeys`・全量面の開く部品）。MUST: `"precomputed"` は umT5 の
 * 部品を開かない — 開くと umT5 の取得（i8 で約 5.3 GiB）が軽い使い方にも乗る（決定 7 の「umT5 を取らない
 * 軽い使い方」が消える）。
 */
const COMPONENT_KEYS: Readonly<Record<WanTextEncoderRoute, readonly string[]>> = {
  gpu: [TRANSFORMER, VAE_DECODER_FIRST, VAE_DECODER_NEXT, TEXT_ENCODER],
  precomputed: [TRANSFORMER, VAE_DECODER_FIRST, VAE_DECODER_NEXT],
};

/**
 * 構築オプションの経路の綴りを読む（省略は既定 `"gpu"`）。未知の綴りは素の `Error`（model / quant 名の
 * 綴り違いと同じ扱い — ADR 0107 決定 3）。MUST: 取得の前に呼ぶ — 経路で取る部品が変わる。
 */
const textEncoderRouteOf = (options: WanPipelineOptions): WanTextEncoderRoute => {
  const route: unknown = options.textEncoder ?? DEFAULT_TEXT_ENCODER;
  if (route !== "gpu" && route !== "precomputed") {
    throw new Error(
      `WanPipeline: textEncoder '${String(route)}' は 'gpu' / 'precomputed' のどちらでもない`,
    );
  }
  return route;
};

/**
 * `"gpu"` の経路で、選んだモデルが umT5 の部品を宣言しているかを見る（取得・容器を開く前）。
 *
 * WHY: 宣言が無いと、取得面・全量面の汎用の文言（「部品 'text_encoder' の容器が無い」）で落ち、
 * 既定が `"gpu"` であることも、umT5 を持たない配布形は `"precomputed"` で読めることも伝わらない。
 * 未知の model はここでは見ない（admission が利用可能な一覧つきで落とす）。
 */
const assertTextEncoderDeclared = (
  manifest: Manifest,
  options: WanPipelineOptions,
  route: WanTextEncoderRoute,
): void => {
  if (route !== "gpu") return;
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) return;
  if (!Object.hasOwn(manifest.models[modelName].weights, TEXT_ENCODER)) {
    throw new Error(
      `WanPipeline: model '${modelName}' の weights に umT5 の部品 '${TEXT_ENCODER}' が無い` +
        '（textEncoder の既定 "gpu" が取る。umT5 を持たない配布形を事前計算の埋め込みで回すなら ' +
        'textEncoder: "precomputed"）',
    );
  }
};

/**
 * テキスト埋め込み資産のキー（段 7 の manifest のモデル単位の `assets` — 決定 4）。両方の経路で読む —
 * `"precomputed"` の受理集合で、`"gpu"` でも {@link WanPipeline.prompts}（例示の一覧）の出所。
 */
const TEXT_EMBEDS = "text_embeds";

/**
 * umT5 のトークナイザ資産のキー（manifest のモデル単位の `assets` — ADR 0119 追記「段 10d の設計」C・
 * 形式 `karume-wan-umt5-tokenizer/1`）。読むのは `"gpu"` の経路だけ。
 */
const UMT5_TOKENIZER = "umt5_tokenizer";

/**
 * negative を省いたときの既定（公式 Wan2.1 の `wan_shared_cfg.sample_neg_prompt` の原文 — 全角の読点を
 * 含めて逐語。前処理が `,` へ畳む）。`"gpu"` の経路はこの文字列を GPU で符号化する（ADR 0119 追記
 * 「段 10d の設計」— 資産の行は使わない: 決定 7 の「positive も negative も GPU で作る」）。
 *
 * MUST: テキスト埋め込み資産の `negative` の行の原文とビット同一（`"precomputed"` の既定と同じ文字列を
 * 指す）。`wan_pipeline_test.ts` が recipe の固定プロンプト（`tools/export-recipes/wan/prompts.py` —
 * fixture `wan-text/parity.json` の `fixed-negative`）と資産の行の両方と突き合わせる。
 *
 * NOTE: `export` はその突き合わせのテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const WAN_DEFAULT_NEGATIVE_PROMPT: string =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，" +
  "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，" +
  "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，" +
  "三条腿，背景人很多，倒着走";

/** `transformer` の容器が宣言する RoPE の素表の資産名（役割 `rope-base`）。 */
const ROPE_BASE = "rope_base";

/** DiT の S 形グラフの入力名（recipe `wan/export_dit.py` の forward の引数名）。 */
const DIT_TOKENS = "tokens";
const DIT_TIMESTEPS_PROJ = "timesteps_proj";
const DIT_CONTEXT = "encoder_hidden_states";
const DIT_ROPE_COS = "rope_cos";
const DIT_ROPE_SIN = "rope_sin";

/**
 * DiT の patch（上流 transformer の config `patch_size [1, 2, 2]`・`in_channels 16` — アーキ定数）。
 * グラフの `tokens` の最終次元（`16·1·2·2 = 64`）と構築時に突き合わせる。
 */
const WAN_PATCH: WanPatchGeometry = {
  channels: 16,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

/** VAE の時間圧縮（最初の chunk は 1 フレーム・以降は 4 フレーム — `vae-chunks.ts` の取り決め）。 */
const TEMPORAL_COMPRESSION = 4;

/**
 * 受理集合（ADR 0118 決定 7 — 832×480 / 480×832 × フレーム数 4n+1 の 5〜81）。検収したのは
 * 832×480 の 33 フレーム（段 3 / 5 / 6）と 81 フレーム（段 8）、480×832 の VAE（段 5）。
 *
 * 81 フレームの可否は DiT 段単独の VRAM だけで決まる — 冒頭の NOTE の実測どおり DiT の Session を
 * 畳んだ直後に確保が戻り、DiT 段と VAE 段は重ならない（切り替えの山 = DiT 段の山）。段 8 の実測
 * （B570・81 フレーム 50 ステップの通し・fdinfo）で DiT 段の山は 7.31 GiB・VAE 段の山は 3.78 GiB
 * （ADR 0118「段 8 の結果」）。MUST: 拒む文言は上限（81）だけを言う（利用者に段の番号は意味を
 * 持たない）。
 *
 * MUST: 変えるときはモデルカード（`tools/export-recipes/wan/card.py` の `WAN_ACCEPTED_SIZES` /
 * `WAN_FRAMES`）と `tests/fixtures/wan-card-limits.json` も同じ値にする — カードは manifest に無い
 * この事実を写しで持つので、fixture を挟んだ両側のテスト（wan_pipeline_test.ts と recipe の
 * test_distribution.py）が片側だけの更新を赤にする。
 *
 * NOTE: 3 つの `export` は fixture との突き合わせのテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const ACCEPTED_SIZES: readonly { readonly width: number; readonly height: number }[] = [
  { width: 832, height: 480 },
  { width: 480, height: 832 },
];
export const MIN_FRAMES = 5;
export const MAX_FRAMES = 81;

/**
 * 生成の既定のうち配布形が宣言しないもの（最初の到達目標の 832×480・33 フレーム — 受理集合の側の
 * 事実）。step 数・guidance・shift の既定は manifest の `pipelineConfig`（{@link WanPipelineConfig}）。
 */
const DEFAULTS = {
  frames: 33,
  width: 832,
  height: 480,
  seed: 0,
} as const;

/** 生成結果。`data` は `[3, frames, height, width]` の f32（値域 `[-1, 1]` — クランプ済み）。 */
export type GeneratedVideo = {
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/** `denoise-step` の `copyLatents()` が返す途中の潜在の写し（`[C, F', H/8, W/8]`）。 */
export type WanLatentSnapshot = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly shape: readonly number[];
};

/**
 * {@link WanPipelineOptions.onRunDiagnostics} が受けるコンポーネント名（Session 1 本 = 1 名）。
 * `text_encoder` は `"gpu"` の経路の umT5（positive / negative の 1 回ずつ）。
 */
export type WanRunComponent =
  | "text_encoder"
  | "transformer"
  | "vae_decoder_first"
  | "vae_decoder_next";

/** {@link WanGenerateRequest.onEvent} が受ける生成イベント。 */
export type WanGenerateEvent =
  /**
   * 段の開始（`start` — Session を張る前）と終了（`end` — Session を畳んだ後）。
   * `text_encoder` は `"gpu"` の経路の umT5 の段（`"precomputed"` では出ない）。`vae_decoder` は
   * chunk グラフ 2 本の段。途中で落ちたら `end` は出ない。
   */
  | {
    readonly kind: "stage";
    readonly component: "text_encoder" | "transformer" | "vae_decoder";
    readonly at: "start" | "end";
  }
  | {
    readonly kind: "denoise-step";
    /** 完了した step 数（1 始まり）。 */
    readonly step: number;
    readonly steps: number;
    /** その step で DiT へ渡した timestep（整数）。 */
    readonly timestep: number;
    /** 呼んだときだけ途中の潜在を写して返す（UniPC の更新の後の潜在）。 */
    readonly copyLatents: () => WanLatentSnapshot;
  }
  /** VAE のタイル 1 枚の decode の完了（`tile` は 1 始まり）。 */
  | { readonly kind: "vae-tile"; readonly tile: number; readonly tiles: number };

/**
 * 1 回の生成要求。省いた step 数・guidance・shift は manifest の `pipelineConfig` の既定（配布形の値は
 * 参照の設定 — 50 ステップ・guide 5.0・shift 3.0）、寸法とフレーム数は 832×480・33 フレーム。
 */
export type WanGenerateRequest = {
  /**
   * プロンプト。受理集合は構築時の経路（{@link WanPipelineOptions.textEncoder}）で決まる:
   *
   * - `"gpu"`（既定）: 任意の文字列。上流の前処理（`prompt_clean`）の鏡像とトークナイザの門を通り、
   *   通らない入力は `ModelInputError`（ADR 0119 決定 1・2・4）— 語彙外の文字・Unicode 16.0.0 で
   *   未割り当ての文字・C1 制御文字・HTML の文字参照になりうる並び（`R&D` は `R & D` のように空白を挟めば
   *   通る）・文字化けに見える並び・本文中の特殊トークン・空白の直後の `▁`・空や空白だけ（1 トークン）・
   *   512 トークン超。拒否の文言は直し方まで言う。
   * - `"precomputed"`: **テキスト埋め込み資産の集合にある文字列だけ**（原文か正規化後の文字列の
   *   どちらかに完全一致 — {@link WanPipeline.prompts}）。集合の外は `ModelInputError`（ADR 0118
   *   決定 4）。
   */
  readonly prompt: string;
  /**
   * CFG の uncond 側のプロンプト（{@link WanGenerateRequest.prompt} と同じ受理集合）。省くと公式の
   * `sample_neg_prompt`（`"gpu"` は GPU で符号化した既定の文字列・`"precomputed"` は資産の `negative` の
   * 行 — 同じ原文）。`guidance` が 1（CFG を回さない）のときに渡すと `ModelInputError`（効かないノブを
   * 黙って受けない）。
   */
  readonly negativePrompt?: string;
  /** 初期ノイズの seed（既定 0 — {@link WanGenerateRequest.latents} とは排他）。 */
  readonly seed?: number;
  /**
   * 初期ノイズを外から渡す（`[16, F', H/8, W/8]` の f32・`F' = (frames − 1)/4 + 1`）。参照との照合で
   * torch の `randn` の列を注入する口（seed の生成器は torch とは別の列 — `random.ts`）。書き換えない。
   */
  readonly latents?: Float32Array<ArrayBuffer>;
  /**
   * denoise の step 数（1 以上・既定は `pipelineConfig.defaults.steps`）。`shift` との組で σ 列が
   * 狭義単調減少にならない値（数十万 step など）は `ModelInputError`。
   */
  readonly steps?: number;
  /**
   * CFG の強さ（1 以上で f32 に丸めても有限・既定は `pipelineConfig.defaults.guidance`）。1 なら
   * uncond 側を回さない（上流の `guidance_scale > 1` の判定と同じ）。1 未満は上流が CFG ごと切って値が
   * 効かないので `ModelInputError`。
   */
  readonly guidance?: number;
  /**
   * flow matching の shift（正・既定は `pipelineConfig.scheduler.shift`）。`steps` との組で σ 列が
   * 壊れる値（極端に大きい / 小さい shift）は `ModelInputError`。
   */
  readonly shift?: number;
  /** フレーム数（4n+1 の 5〜81・既定 33）。 */
  readonly frames?: number;
  /** 幅 × 高さ（832×480 か 480×832・既定 832×480）。 */
  readonly width?: number;
  readonly height?: number;
  /**
   * 生成イベントの観測席（{@link WanGenerateEvent}）。**await する**（発火の順が決定的になる）。
   * 例外は握らない — throw が step 粒度の中断になる（段の Session は畳まれる）。
   *
   * MUST: `onEvent` の中で同じパイプラインの `generate` / `dispose` を await しない（直列化鎖の
   * 自己デッドロック）。
   */
  readonly onEvent?: (event: WanGenerateEvent) => void | Promise<void>;
  /**
   * 生成の中断。段の境目（text → DiT → VAE）・各 run の前（text の positive / negative・DiT の各 step）・
   * VAE のタイルの間で、イベントループへ 1 度譲ってから検査する。1 回の run は不可分で、中断は次の
   * 境目で効く。中断したら開いている Session を畳んでから `signal.reason` を**そのまま**投げる（包まない
   * — `error === controller.signal.reason` で自分の中断を識別できる）。中断の後も同じパイプラインで
   * 次の `generate` を回せる。
   */
  readonly signal?: AbortSignal;
};

/** 構築オプション（{@link WanPipeline.fromAssets} / {@link WanPipeline.fromPretrained} 共通）。 */
export type WanPipelineOptions = {
  /** モデル（manifest の models のキー）。省略時は `defaultModel`。 */
  readonly model?: string;
  /** 実行構成（そのモデルの quants のキー）。省略時は `defaultQuant`。 */
  readonly quant?: string;
  /**
   * テキストエンコーダの経路（既定 `"gpu"` — ADR 0119 決定 7・追記「段 10d の設計」B）。取る部品が
   * 経路で変わるので構築時に決める（生成ごとには切り替えない）。
   *
   * - `"gpu"`: umT5（manifest の部品 `text_encoder` — 配布形では umT5 の配布リポへの越境参照・重み i8 で
   *   約 5.3 GiB）とトークナイザ資産（`umt5_tokenizer`）を取り、`generate` ごとに umT5 の段を張って畳む。
   *   任意のプロンプトを受ける。手元の配布形（`localDirectory` / `denoDirectory`）では、越境先を
   *   取得元の `crossRepo` の mapping で渡す（隣のディレクトリを推測しない — hub の `local.ts`）。
   * - `"precomputed"`: umT5 を取らず（1 バイトも読まない）、テキスト埋め込み資産のプロンプトだけを受ける
   *   軽い使い方。資産の埋め込みは bf16 の umT5 の出力で、`"gpu"`（i8 の重み・f32 の活性）とは値が
   *   一致しない — 同じ文字列でも経路を変えれば別の動画になる。
   *
   * 未知の綴りは素の `Error`（取得の前に落ちる）。
   */
  readonly textEncoder?: "gpu" | "precomputed";
  /**
   * 既存の GPU を共有する（渡した側が所有権を持つ — {@link WanPipeline.dispose} は破棄しない）。
   * 省くとパイプラインが `acquireGpu` し、`dispose` で破棄する。
   *
   * MUST: 計測（`acquireGpu({ gpuTiming: true })`）の device は渡せない — VAE の段は 1 タイル = 1 batch
   * で回り、runtime は計測の device で batch を開かない（`GpuContext.beginBatch` の MUST）。構築時に
   * 落とす（DiT の段を回し終えてから VAE の段で落ちる形にしない）。DiT の GPU 時間は DiT だけを回す
   * e2e（段 3）で測る。
   */
  readonly gpu?: GpuContext;
  /**
   * Session の診断の観測席（umT5 は 1 プロンプト = 1 回・DiT は 1 forward = 1 回・VAE は 1 タイル =
   * first / next の 1 回ずつ）。例外は握らない（fail loudly）。
   */
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
  /**
   * 構築の中断。{@link WanPipeline.fromPretrained} は同じ 1 本を取得層へも渡すので、取得（umT5 だけで
   * 約 5.3 GiB）と組み立てのどちらの最中でも効く。組み立ての段の境目（入口・容器を開いた後・資産の
   * 解析の前・GPU の取得の前後）でイベントループへ 1 度譲ってから検査する。中断の例外は
   * `signal.reason` を**そのまま**投げる（包まない — anima / irodori の構築と同じ形）。
   *
   * NOTE: 構築の `signal` は生成には持ち越さない（生成の中断は {@link WanGenerateRequest.signal}）—
   * 構築の寿命の値（`AbortSignal.timeout` など）が以後の生成を全部落とす形にしない。
   */
  readonly signal?: AbortSignal;
};

/**
 * {@link WanPipeline.fromPretrained} が追加で受ける取得層のオプション（hub へ透過する）。
 *
 * NOTE: `headers` / `fetch` / `caches` / `onRetry` が **HTTP 取得元専用**であることを含め、欄ごとの
 * 説明は {@link FromPretrainedHubOptions} に 1 本化してある。
 */
export type WanFromPretrainedOptions =
  & WanPipelineOptions
  & FromPretrainedHubOptions
  & FromPretrainedComponentOptions;

/**
 * 取得済みの manifest + 資産（hub の `fetchAssets` の返り値をそのまま渡せる形）。`assets` の部品は
 * 単一形 `krm` のキー（`transformer`）か part 列（`transformer[0]` / `transformer[1]` / … — part 0 から
 * 添字順）で、`transformer` / `vae_decoder_first` / `vae_decoder_next` の 3 本と、`"gpu"` の経路
 * （既定）では `text_encoder`（umT5）。加えて manifest の `assets` の `text_embeds`（埋め込み資産の
 * safetensors — 両方の経路）と、`"gpu"` の経路では `umt5_tokenizer`（トークナイザ資産の JSON）。
 */
export type WanAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/** 初期ノイズの出所（{@link planWanGeneration}）。 */
type InitialNoise =
  | { readonly kind: "seed"; readonly seed: number }
  | { readonly kind: "latents"; readonly data: Float32Array<ArrayBuffer> };

/**
 * 入口の門を通った生成の要求（{@link planWanGeneration} / {@link planWanGpuGeneration} の戻り）。
 * `Text` はプロンプト 1 本の形（`"precomputed"` = 資産の埋め込みの行・`"gpu"` = umT5 の id 列）。
 */
export type WanGenerationPlan<Text = WanTextEmbedding> = WanGenerationKnobs & {
  readonly positive: Text;
  /** CFG の uncond 側（`guidance` が 1 なら undefined — uncond を回さない）。 */
  readonly negative: Text | undefined;
};

/** 計画のうちプロンプト以外（DiT と VAE の段が使う — 経路に依らない）。 */
type WanGenerationKnobs = {
  readonly steps: number;
  readonly guidance: number;
  readonly shift: number;
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  /** 潜在の形 `[16, F', H/8, W/8]`。 */
  readonly latentShape: readonly [number, number, number, number];
  readonly initial: InitialNoise;
  /** `steps` × `shift` の UniPC の σ 列と timestep 列（denoise はこれを使い、組み直さない）。 */
  readonly schedule: WanUniPcSchedule;
  /** VAE のタイル計画（潜在 `H/8 × W/8` — denoise の前に立てる）。 */
  readonly tiles: WanVaeTilePlan;
};

/** 計画に要る VAE の chunk グラフの幾何（資産の宣言から — {@link wanVaeChunkLayout}）。 */
type PlanLayout = Pick<WanVaeChunkLayout, "latentChannels" | "tile" | "sampleTile">;

/** 経路ごとのプロンプトの門（{@link planWith} が 1 本の検査の順で呼ぶ）。 */
type PromptGate<Text> = {
  /** 渡された文字列（型は検査済み）を受理集合で引く / 符号化する。拒否は `ModelInputError`。 */
  readonly resolve: (text: string, what: "prompt" | "negativePrompt") => Text;
  /** `negativePrompt` を省いたときの既定（`guidance` > 1 のときだけ呼ぶ）。 */
  readonly defaultNegative: () => Text;
};

/**
 * 生成の要求を検査して計画にする（`generate` の入口・GPU に触る前の純粋な門）。省いた step 数・
 * guidance・shift は `config`（manifest の `pipelineConfig`）の既定で埋める。`"precomputed"` の経路の
 * 入口で、プロンプトはテキスト埋め込み資産の集合で引く。
 *
 * MUST: 入力起因の失敗（集合の外のプロンプト・受理集合の外の寸法・値域外のノブ・σ 列が壊れる
 * steps × shift の組）は全部ここで `ModelInputError` にする。DiT の重みを上げてから落ちる形にしない。
 * 既定だけの組は manifest の門（`parseWanPipelineConfig`）が σ 列まで見ているので、ここで σ 列が
 * 壊れるのは要求が値を渡したときだけ（= 入力起因）。
 *
 * MUST: VAE のタイル計画もここで立てる — denoise の後に立てると、資産だけで判る不整合が全 step を
 * 払った後に落ちる。受理する寸法は家族 admission（{@link assertWanVaeTilesCoverAcceptedSizes}）が
 * 全数を通しているので、ここで落ちるのは admission を経ない呼び出しだけ（資産の齟齬 — 素の `Error`）。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const planWanGeneration = (
  request: WanGenerateRequest,
  embeds: WanTextEmbeds,
  layout: PlanLayout,
  config: WanPipelineConfig,
): WanGenerationPlan<WanTextEmbedding> =>
  planWith(
    request,
    {
      resolve: (text, what) => {
        const entry = findWanTextEmbedding(embeds, text);
        if (entry === undefined) {
          throw new ModelInputError(
            `${what} がテキスト埋め込み資産の集合に無い（事前計算の経路〈textEncoder: "precomputed"〉は ` +
              `${embeds.entries.length} 本だけを受ける: ${
                embeds.entries.map((entry) => entry.name).join(" / ")
              } — 原文か正規化後の文字列に完全一致させる。WanPipeline.prompts で引ける）`,
          );
        }
        return entry;
      },
      defaultNegative: () => {
        const defaults = embeds.entries.filter((entry) => entry.role === "negative");
        if (defaults.length !== 1) {
          throw new ModelInputError(
            `negativePrompt を省いたが、資産に negative の行が ${defaults.length} 本ある（1 本のときだけ既定にする）`,
          );
        }
        return defaults[0];
      },
    },
    layout,
    config,
  );

/**
 * `"gpu"` の経路の入口（{@link planWanGeneration} と同じ検査の順・同じノブの門）。プロンプトは umT5 の
 * プロンプト層（`prompt_clean` の鏡像 → トークナイザ — {@link WanPromptEncoder.encode}）で id 列に
 * する。拒否は `ModelInputError`（前処理の拒否は派生の `PromptCleanError`）で、文言は直し方まで言う。
 * `negativePrompt` を省くと {@link WAN_DEFAULT_NEGATIVE_PROMPT} を同じ門で符号化する。
 *
 * MUST: 符号化はここで済ませる — umT5 の重み（i8 で約 5.3 GiB）を上げてから語彙外で落ちる形にしない。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const planWanGpuGeneration = (
  request: WanGenerateRequest,
  encoder: WanPromptEncoder,
  layout: PlanLayout,
  config: WanPipelineConfig,
): WanGenerationPlan<Int32Array<ArrayBuffer>> =>
  planWith(
    request,
    {
      resolve: (text, what) => encoder.encode(text, what),
      defaultNegative: () => encoder.encode(WAN_DEFAULT_NEGATIVE_PROMPT, "negativePrompt（既定）"),
    },
    layout,
    config,
  );

/** 2 つの経路の入口が共有する検査の本体（プロンプトの引き方だけを `gate` から受ける）。 */
const planWith = <Text>(
  request: WanGenerateRequest,
  gate: PromptGate<Text>,
  layout: PlanLayout,
  config: WanPipelineConfig,
): WanGenerationPlan<Text> => {
  const resolve = (text: unknown, what: "prompt" | "negativePrompt"): Text => {
    if (typeof text !== "string") throw new ModelInputError(`${what} が文字列でない`);
    return gate.resolve(text, what);
  };
  const positive = resolve(request.prompt, "prompt");

  const steps = request.steps ?? config.defaults.steps;
  if (!Number.isInteger(steps) || steps < 1) {
    throw new ModelInputError(`steps ${steps} が 1 以上の整数でない`);
  }
  const guidance = request.guidance ?? config.defaults.guidance;
  // f32 で見る: CFG は `f32(guidance)` で掛ける（`wanClassifierFreeGuidance`）ので、f64 で有限でも
  // f32 で Infinity になる値（`Number.MAX_VALUE` など）は合成を NaN にする。
  if (!Number.isFinite(Math.fround(guidance)) || guidance < 1) {
    throw new ModelInputError(
      `guidance ${guidance} が 1 以上で f32 に収まる有限の数でない` +
        "（上流は 1 以下で CFG を回さないので、1 未満は効かない・CFG は f32 で掛ける）",
    );
  }
  const shift = request.shift ?? config.scheduler.shift;
  if (!Number.isFinite(shift) || shift <= 0) {
    throw new ModelInputError(`shift ${shift} が正の有限の数でない`);
  }
  let schedule: WanUniPcSchedule;
  try {
    schedule = wanUniPcSchedule(steps, shift, WAN_UNIPC_CONFIG.numTrainTimesteps);
  } catch (error) {
    // 単項の門を通った steps / shift の組で σ 列が壊れる（低水準の RangeError — scheduler.ts の
    // MUST）。既定の組は manifest の門が通しているので、ここに来るのは要求が渡した値のとき。
    if (!(error instanceof RangeError)) throw error;
    throw new ModelInputError(
      `steps ${steps} と shift ${shift} の組では UniPC の σ 列が組めない（${error.message}）`,
      { cause: error },
    );
  }

  // CFG の uncond 側（上流の `do_classifier_free_guidance = guidance_scale > 1`）。
  let negative: Text | undefined;
  if (guidance > 1) {
    negative = request.negativePrompt !== undefined
      ? resolve(request.negativePrompt, "negativePrompt")
      : gate.defaultNegative();
  } else if (request.negativePrompt !== undefined) {
    throw new ModelInputError(
      "guidance 1 では uncond 側を回さないので negativePrompt は効かない（効かせるなら guidance を 1 より大きくする）",
    );
  }

  const frames = request.frames ?? DEFAULTS.frames;
  if (
    !Number.isInteger(frames) || frames < MIN_FRAMES || frames > MAX_FRAMES ||
    (frames - 1) % TEMPORAL_COMPRESSION !== 0
  ) {
    throw new ModelInputError(
      `frames ${frames} が受理集合（4n+1 の ${MIN_FRAMES}〜${MAX_FRAMES}）に無い`,
    );
  }
  const width = request.width ?? DEFAULTS.width;
  const height = request.height ?? DEFAULTS.height;
  if (!ACCEPTED_SIZES.some((size) => size.width === width && size.height === height)) {
    throw new ModelInputError(
      `${width}×${height} が受理集合（${
        ACCEPTED_SIZES.map((size) => `${size.width}×${size.height}`).join(" / ")
      }）に無い`,
    );
  }
  const spatialScale = layout.sampleTile / layout.tile;
  const latentShape: [number, number, number, number] = [
    WAN_PATCH.channels,
    (frames - 1) / TEMPORAL_COMPRESSION + 1,
    height / spatialScale,
    width / spatialScale,
  ];
  if (!latentShape.every(Number.isInteger)) {
    // 受理集合は資産の縮尺で割り切れる寸法だけ — 割れるなら資産の取り違え（入力起因ではない）。
    throw new Error(
      `潜在の形 [${latentShape}] が整数でない（VAE の縮尺 ${spatialScale}）`,
    );
  }

  if (request.seed !== undefined && request.latents !== undefined) {
    throw new ModelInputError("seed と latents は排他（初期ノイズの出所はどちらか 1 つ）");
  }
  let initial: InitialNoise;
  if (request.latents !== undefined) {
    const { latents } = request;
    const count = latentShape.reduce((product, dim) => product * dim, 1);
    if (!(latents instanceof Float32Array) || latents.length !== count) {
      throw new ModelInputError(
        `latents が Float32Array [${latentShape.join(", ")}]（${count} 要素）でない`,
      );
    }
    if (!latents.every(Number.isFinite)) throw new ModelInputError("latents に非有限値がある");
    initial = { kind: "latents", data: latents };
  } else {
    const seed = request.seed ?? DEFAULTS.seed;
    assertAcceptableSeed(seed);
    initial = { kind: "seed", seed };
  }

  const tiles = planWanVaeTiles(layout, latentShape[2], latentShape[3]);
  return {
    positive,
    negative,
    steps,
    guidance,
    shift,
    frames,
    width,
    height,
    latentShape,
    initial,
    schedule,
    tiles,
  };
};

/**
 * VAE の chunk グラフの幾何で、受理する寸法（{@link ACCEPTED_SIZES}）が全部タイルで覆えることを
 * 見る（家族 admission の門 — 寸法は有限で 2 通りなので全数を計画する）。
 *
 * MUST: admission で呼ぶ。タイル辺は配布物だけで差し替えられる（ADR 0118 決定 2）ので、潜在の短辺
 * より大きいタイル（例 64 > 60）や重なりの下限を満たせないタイル（例 8）の資産も chunk グラフの
 * 検査（{@link wanVaeChunkLayout}）は通る。ここで落とさないと、DiT の段を全部払った後の VAE の段で
 * 初めて落ちる。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const assertWanVaeTilesCoverAcceptedSizes = (layout: PlanLayout): void => {
  const scale = layout.sampleTile / layout.tile;
  for (const { width, height } of ACCEPTED_SIZES) {
    const latentHeight = height / scale;
    const latentWidth = width / scale;
    if (!Number.isInteger(latentHeight) || !Number.isInteger(latentWidth)) {
      throw new Error(
        `WanPipeline: VAE の縮尺 ${layout.sampleTile} / ${layout.tile} では ${width}×${height} の潜在が` +
          "整数にならない",
      );
    }
    try {
      planWanVaeTiles(layout, latentHeight, latentWidth);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(
        `WanPipeline: VAE のタイル（潜在 ${layout.tile}）では ${width}×${height}` +
          `（潜在 ${latentHeight}×${latentWidth}）をタイル decode できない — ${reason}`,
        { cause: error },
      );
    }
  }
};

/** グラフの値の宣言（宣言の無い名前は fail loudly）。 */
const valueOf = (
  owner: GraphOwner,
  name: string,
): GraphOwner["graph"]["values"][string] => {
  if (!Object.hasOwn(owner.graph.values, name)) {
    throw new Error(`WanPipeline: グラフの値 '${name}' の宣言が無い`);
  }
  return owner.graph.values[name];
};

/** グラフの値の形（宣言の無い名前は fail loudly）。 */
const valueShape = (owner: GraphOwner, name: string): readonly (number | string)[] =>
  valueOf(owner, name).shape;

/** グラフ入力の形（無ければ fail loudly）。 */
const inputShape = (owner: GraphOwner, name: string): readonly (number | string)[] => {
  const spec = owner.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) throw new Error(`WanPipeline: transformer のグラフ入力 '${name}' が無い`);
  return spec.shape;
};

/** 静的次元（記号次元なら fail loudly）。 */
const staticDim = (dims: readonly (number | string)[], axis: number, where: string): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") throw new Error(`WanPipeline: ${where} の軸 ${axis} が静的でない`);
  return dim;
};

/**
 * 宣言の形を rank と全軸で照合する（数は静的次元・文字列は記号次元の名前）。`expected` はホストが
 * 組む形そのもの。
 */
const assertDims = (
  dims: readonly (number | string)[],
  expected: readonly (number | string)[],
  where: string,
): void => {
  if (dims.length !== expected.length || dims.some((dim, axis) => dim !== expected[axis])) {
    const declared = dims.join(", ");
    const host = expected.join(", ");
    throw new Error(`WanPipeline: ${where} の形 [${declared}] がホストの組む [${host}] と違う`);
  }
};

/** 構築時に確かめた DiT のグラフの取り決め。 */
type DitContract = {
  /** グラフ出力の名前（`[1, S, 64]`）。 */
  readonly output: string;
  /** `timesteps_proj [1, W]` の W。 */
  readonly projWidth: number;
  /** `encoder_hidden_states [1, rows, width]` の rows（ゼロで埋める先の行数）。 */
  readonly contextRows: number;
  readonly contextWidth: number;
};

/**
 * DiT のグラフ宣言を、ホストが組む入力（patch・RoPE の素表）と突き合わせる。
 *
 * MUST: 構築時に落とす。ホストの前処理は自分の定数で組むので、グラフが別の寸法で焼かれていても
 * ホスト側は最後まで通り、落ちるのは DiT の重みを上げた後の Session の shape 検査になる。取得面では
 * 家族 admission（重みの part を取る前）で呼ぶ — 埋め込み資産との突合（{@link assertEmbedsFitContext}）は
 * 資産のバイト列が届いてから。
 *
 * MUST: 最終次元だけでなく rank・batch（B = 1 — 決定 5）・可変の S まで照合する。ホストは
 * `tokens [1, S, 64]`・`rope_cos / rope_sin [1, S, 1, w]` を組み、出力を `[1, S, 64]` として読む
 * （{@link WanPipeline.#denoise}）ので、batch 2・固定の S・rank 違いの宣言も Session の shape 検査まで
 * 通ってしまう。S は寸法とフレーム数ごとに変わるので記号次元で、4 本とも**同じ記号**であること
 * （IR の記号は上下限を持たないので、上限の突合は要らない）。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const ditContract = (transformer: GraphOwner, ropeBase: WanRopeBase): DitContract => {
  const tokenWidth = wanTokenWidth(WAN_PATCH);
  const tokens = inputShape(transformer, DIT_TOKENS);
  const sequence = tokens.at(1);
  if (typeof sequence !== "string") {
    throw new Error(
      `WanPipeline: '${DIT_TOKENS}' の軸 1 が記号次元でない（${String(sequence)}）— ` +
        "ホストは S を寸法とフレーム数ごとに変えて渡す",
    );
  }
  assertDims(tokens, [1, sequence, tokenWidth], `'${DIT_TOKENS}'`);
  const [output] = transformer.graph.outputs;
  if (transformer.graph.outputs.length !== 1) {
    throw new Error(
      `WanPipeline: transformer の出力が ${transformer.graph.outputs.length} 本（1 本の S 形）`,
    );
  }
  assertDims(
    valueShape(transformer, output),
    [1, sequence, tokenWidth],
    `transformer の出力 '${output}'`,
  );
  const ropeWidth = wanRopeWidth(ropeBase);
  for (const name of [DIT_ROPE_COS, DIT_ROPE_SIN]) {
    assertDims(inputShape(transformer, name), [1, sequence, 1, ropeWidth], `'${name}'`);
  }
  const proj = inputShape(transformer, DIT_TIMESTEPS_PROJ);
  const projWidth = staticDim(proj, 1, DIT_TIMESTEPS_PROJ);
  assertDims(proj, [1, projWidth], `'${DIT_TIMESTEPS_PROJ}'`);
  const context = inputShape(transformer, DIT_CONTEXT);
  const contextRows = staticDim(context, 1, DIT_CONTEXT);
  const contextWidth = staticDim(context, 2, DIT_CONTEXT);
  assertDims(context, [1, contextRows, contextWidth], `'${DIT_CONTEXT}'`);
  return { output, projWidth, contextRows, contextWidth };
};

/** 構築時に確かめた umT5 のグラフの取り決め（`"gpu"` の経路）。 */
type Umt5Contract = {
  /** グラフ出力の名前（`[1, L, width]` — width は DiT の文脈の幅）。 */
  readonly output: string;
};

/** umT5 のグラフ入力の宣言（無ければ fail loudly）。 */
const umt5Input = (
  textEncoder: GraphOwner,
  name: string,
): GraphOwner["graph"]["inputs"][number] => {
  const spec = textEncoder.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    const declared = textEncoder.graph.inputs.map((input) => input.name).join(" / ");
    throw new Error(
      `WanPipeline: text_encoder のグラフ入力 '${name}' が無い（宣言: ${declared}）`,
    );
  }
  return spec;
};

/**
 * umT5 のグラフ宣言を、ホストが組む入力（id 列 `[1, L]` とバケット表 `[L, L]` の i32 — `umt5/session-io.ts`）
 * と DiT の文脈入力に突き合わせる（ADR 0119 決定 3・4・5）。
 *
 * MUST: 家族 admission（重みの part を取る前）で呼ぶ。ホストは自分の定数で入力を組み、出力を
 * `[1, L, contextWidth]` として DiT へ詰める（{@link padUmt5Context}）ので、宣言が違っていても落ちるのは
 * umT5 の重み（i8 で約 5.3 GiB）を上げた後の Session の shape 検査か、詰めの検査になる。
 *
 * - 入力はちょうど 2 本（`input_ids` `[1, L]`・`relative_position_buckets` `[L, L]`・どちらも i32）で、
 *   L は**同じ 1 つの記号次元**（有効長ごとに形を変えて渡す — 決定 4）。
 * - 出力は 1 本の f32 `[1, L, W]` で、L は入力と同じ記号・W は DiT の `encoder_hidden_states` の幅。
 * - 有効長の上限（{@link WAN_UMT5_MAX_LENGTH}）が DiT の文脈の行数に収まる。
 * - 格納は i8（`int8-sym` = per-channel）の重みと f32 の表だけ（決定 5 — GPU の移植の門で検証した
 *   組み合わせ。text 段の Session は quant の宣言を受けない〈`{}`〉ので、格納が実行の形を決める）。
 *   i8 を 1 本も持たない容器（全部 f32 など）も受けない。
 *
 * NOTE: `export` は GPU 無しで門を縛るテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const umt5Contract = (
  textEncoder: GraphOwner,
  dit: Pick<DitContract, "contextRows" | "contextWidth">,
): Umt5Contract => {
  const { graph } = textEncoder;
  const ids = umt5Input(textEncoder, WAN_UMT5_INPUT_IDS);
  const buckets = umt5Input(textEncoder, WAN_UMT5_RELATIVE_POSITION_BUCKETS);
  if (graph.inputs.length !== 2) {
    throw new Error(
      `WanPipeline: text_encoder のグラフ入力が ${graph.inputs.length} 本（` +
        `'${WAN_UMT5_INPUT_IDS}' と '${WAN_UMT5_RELATIVE_POSITION_BUCKETS}' の 2 本だけを組む）`,
    );
  }
  const length = ids.shape.at(1);
  if (typeof length !== "string") {
    throw new Error(
      `WanPipeline: text_encoder の '${WAN_UMT5_INPUT_IDS}' の軸 1 が記号次元でない` +
        `（${String(length)}）— ホストは有効長 L ごとに形を変えて渡す`,
    );
  }
  for (const [spec, expected] of [[ids, [1, length]], [buckets, [length, length]]] as const) {
    if (spec.dtype !== "i32") {
      throw new Error(
        `WanPipeline: text_encoder の '${spec.name}' の dtype ${spec.dtype} が i32 でない`,
      );
    }
    assertDims(spec.shape, expected, `text_encoder の '${spec.name}'`);
  }
  if (graph.outputs.length !== 1) {
    throw new Error(
      `WanPipeline: text_encoder の出力が ${graph.outputs.length} 本（1 本の [1, L, W]）`,
    );
  }
  const [output] = graph.outputs;
  const value = valueOf(textEncoder, output);
  if (value.dtype !== "f32") {
    throw new Error(
      `WanPipeline: text_encoder の出力 '${output}' の dtype ${value.dtype} が f32 でない`,
    );
  }
  assertDims(value.shape, [1, length, dit.contextWidth], `text_encoder の出力 '${output}'`);
  if (WAN_UMT5_MAX_LENGTH > dit.contextRows) {
    throw new Error(
      `WanPipeline: umT5 の有効長の上限 ${WAN_UMT5_MAX_LENGTH} が DiT の文脈の行数 ${dit.contextRows} を超える`,
    );
  }
  let quantized = 0;
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.storage === undefined) {
      // 共有の宣言（貸し手の Session の重みを借りる — 格納を持たない）。text 段に貸し手は居ない。
      throw new Error(
        `WanPipeline: text_encoder の initializer '${name}' が共有の宣言（text 段は重みを借りない）`,
      );
    }
    const layout = codecLayout(initializer.storage.codec);
    if (layout === "i8") quantized += 1;
    else if (layout !== "f32") {
      throw new Error(
        `WanPipeline: text_encoder の initializer '${name}' の格納 ${initializer.storage.codec} は受けない` +
          "（受けるのは i8 per-channel の重みと f32 の表だけ — ADR 0119 決定 5）",
      );
    }
  }
  if (quantized === 0) {
    throw new Error(
      "WanPipeline: text_encoder に i8 の重みが 1 本も無い（受けるのは i8 per-channel の重みと f32 の表 — " +
        "ADR 0119 決定 5）",
    );
  }
  return { output };
};

/** 埋め込み資産の幅と有効長が DiT の文脈入力 `[1, rows, width]` に収まることを見る。 */
const assertEmbedsFitContext = (dit: DitContract, embeds: WanTextEmbeds): void => {
  if (dit.contextWidth !== embeds.width) {
    throw new Error(
      `WanPipeline: '${DIT_CONTEXT}' の幅 ${dit.contextWidth} が埋め込み資産の幅 ${embeds.width} と違う`,
    );
  }
  const longest = Math.max(...embeds.entries.map((entry) => entry.tokens));
  if (longest > dit.contextRows) {
    throw new Error(
      `WanPipeline: 埋め込みの有効長 ${longest} が文脈の行数 ${dit.contextRows} を超える`,
    );
  }
};

const asF32 = (tensor: Tensor, where: string): Float32Array => {
  if (tensor.dtype !== "f32") throw new Error(`${where}: f32 でない（${tensor.dtype}）`);
  return tensor.data;
};

/** 最初の非有限値の添字（無ければ -1）。 */
const firstNonFinite = (values: Float32Array): number =>
  values.findIndex((value) => !Number.isFinite(value));

/**
 * この家族が manifest の `session` と明示指定で受けるキー（受理表 — `session/options.ts`）。
 *
 * linear / 融合 attention の実行形 3 欄を受ける: 配布形の実用席 `f16+dit8-a8-attn8-s16` がこの 3 つを
 * 宣言する（ADR 0120 決定 1 / 7 — anima の同名の席と同じ宣言）。並列 GEMV と融合の 4 欄は受けない —
 * Wan の DiT のグラフでは効く席も数値も確かめていない組合せだから。受けないキーの宣言は重みを取る
 * 前に fail loudly（黙って既定で走らせない）。
 *
 * 席の選択は既存の `quant` だけで、利用者の明示指定の口は公開面に足さない（ADR 0120 決定 7 —
 * `resolveSessionOptions` の第 3 引数は空のまま。受理表は同じ 1 本なので、要求が出たら後から足せる）。
 *
 * NOTE: `export` は全家族の受理表の網羅を縛るテストのため（`mod.ts` / サブパス面には出さない —
 * ADR 0008）。
 */
export const WAN_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  attentionCompute: true,
  attentionScoreStorage: true,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  packedStaticQuantize: false,
};

/** 家族 admission（`WanPipeline.#admit`）が確定させる材料。 */
type WanAdmission = {
  readonly config: WanPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /** quant 宣言を受理表で通した実効設定（{@link WAN_SESSION_POLICY} — DiT の Session へ渡す）。 */
  readonly sessionOptions: SessionOptions;
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
  readonly layout: WanVaeChunkLayout;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  /** 経路（`"gpu"` は umT5 のグラフの取り決めを伴う）。 */
  readonly textEncoder:
    | { readonly route: "precomputed" }
    | { readonly route: "gpu"; readonly contract: Umt5Contract };
};

/** text 段の材料（経路ごと）。 */
type WanTextStage =
  /** 資産の埋め込みを引く（{@link WanState.textEmbeds} — Session を張らない）。 */
  | { readonly kind: "precomputed" }
  | {
    readonly kind: "gpu";
    /** プロンプト層（前処理 → トークナイザ — 入口の門）。 */
    readonly encoder: WanPromptEncoder;
    /** umT5 の部品（generate ごとに Session を張って畳む — 決定 11）。 */
    readonly component: ModelComponent;
    readonly contract: Umt5Contract;
  };

/** {@link WanPipeline} の内部状態。 */
type WanState = {
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly config: WanPipelineConfig;
  readonly sessionOptions: SessionOptions;
  readonly transformer: ModelComponent;
  readonly vaeFirst: ModelComponent;
  readonly vaeNext: ModelComponent;
  readonly layout: WanVaeChunkLayout;
  readonly ropeBase: WanRopeBase;
  readonly dit: DitContract;
  /** 埋め込み資産（`"precomputed"` の受理集合・両経路の {@link WanPipeline.prompts}）。 */
  readonly textEmbeds: WanTextEmbeds;
  readonly text: WanTextStage;
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/** DiT の文脈入力の中身（`[rows, width]` へ詰めた positive と、CFG の uncond 側）。 */
type WanContexts = {
  readonly positive: Float32Array<ArrayBuffer>;
  readonly negative: Float32Array<ArrayBuffer> | undefined;
};

/**
 * Wan2.1 のテキスト → 動画パイプライン。
 *
 * 構築は {@link WanPipeline.fromPretrained}（配布形から取得）か {@link WanPipeline.fromAssets}（取得済み
 * バイト列）だけを入口にする（コンストラクタは private — manifest 検査と資産の突合を迂回した半端な
 * 状態を作らせない。ADR 0008）。
 */
export class WanPipeline {
  readonly #state: WanState;
  /** generate と dispose の直列化鎖。 */
  readonly #chain = createOperationChain();
  /** dispose の 1 本（undefined でないことが「dispose 済み」）。 */
  #disposal: Promise<void> | undefined;

  private constructor(state: WanState) {
    this.#state = state;
  }

  /**
   * 配布形から取得して組む（`loadManifest` → `resolveSelection` → **各部品の descriptor（part 0）
   * だけ**を取って admission → 重みの part を温める → 埋め込み資産・トークナイザ資産の `fetchAssets` →
   * 構築）。block は Session を組むその瞬間に part 順で読まれる（ADR 0109 — `src/hub/components.ts`）。
   * 取る部品は経路（{@link WanPipelineOptions.textEncoder}）で決まる — `"precomputed"` は umT5
   * （`text_encoder`）を取らない。部品を別リポの同じ役割で差し替えるときは
   * {@link FromPretrainedComponentOptions.components}（`transformer` / `vae_decoder_first` /
   * `vae_decoder_next`、`"gpu"` の経路では `text_encoder` も）。
   *
   * **`ref` は必須**（取得元に既定は無い — `src/hub/repo-ref.ts` の MUST。この家族は公開配布リポを
   * まだ持たないので pin 定数も無い）。文字列の `ref` は `{ repo }` と読む（= `main` 追従）。手元の
   * 配布形（`models/karume-wan2.1`）は**取得元ハンドル**で渡す（`localDirectory` / `@karume/hub/deno` の
   * `denoDirectory`）— HF の `owner/name` の綴りの門は通らず、network も CacheStorage も通らない
   * （{@link WanFromPretrainedOptions} の HTTP 専用ノブは効かない）。umT5 の越境先（umT5 の配布リポ）は
   * その取得元の `crossRepo` に mapping で渡す（例 `denoDirectory("models/karume-wan2.1", { crossRepo:
   * { "<owner>/karume-umt5-xxl": denoDirectory("models/karume-umt5-xxl") } })` — キーは manifest の
   * `text_encoder` が宣言する repo）。
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: WanFromPretrainedOptions = {},
  ): Promise<WanPipeline> {
    // 経路は取得の前に決める（取る部品が変わる）— 綴り違いは 1 バイトも取らずに落とす。
    const route = textEncoderRouteOf(options);
    const source = toManifestSource(ref, "WanPipeline.fromPretrained");
    // signal は取得層と構築の**両方**へ渡す（DL が終わった瞬間に中断が効かなくなる窓を作らない —
    // `hubLoadOptions` が写す・anima / irodori と同じ形）。
    const hubOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    assertTextEncoderDeclared(loaded.manifest, options, route);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    const selection = resolveSelection(loaded.manifest, choice);
    const buildOptions: WanPipelineOptions = {
      ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
      ...choice,
      textEncoder: route,
      ...(options.onRunDiagnostics === undefined
        ? {}
        : { onRunDiagnostics: options.onRunDiagnostics }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    };
    // 家族の門は admission 席で通す（重みの part を取る前 — `src/hub/components.ts`）。
    const { admitted, assets, open } = await loadContainerComponents(
      "WanPipeline.fromPretrained",
      loaded,
      selection,
      COMPONENT_KEYS[route],
      async (open) => {
        const admitted = await WanPipeline.#admit(loaded.manifest, open, buildOptions);
        // 配布形が宣言した `requiredLimits` は**重みの part を取る前**にここで見る
        // （ADR 0089 決定 5 — 共有 GPU ならその limits、自前で取る経路はアダプタ実測値）。
        await assertRequiredLimitsBeforeDownload(
          admitted.quant.requiredLimits,
          buildOptions.gpu,
          `WanPipeline: quant '${admitted.quantName}'`,
        );
        return admitted;
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return await WanPipeline.#build(admitted, assets, open, buildOptions);
  }

  /**
   * 取得済みの manifest + 資産から組む（{@link WanAssets}）。容器を開いて、取得面と同じ家族 admission
   * （{@link WanPipeline.#admit}）と組み立て（{@link WanPipeline.#build}）を通す — 2 面の違いは部品の
   * 供給口だけ。
   */
  static async fromAssets(
    input: WanAssets,
    options: WanPipelineOptions = {},
  ): Promise<WanPipeline> {
    const route = textEncoderRouteOf(options);
    // 入口の検査（中断済みなら容器を 1 本も開かない — 開くのは GB 級の part 列の検証を含む）。
    await settleAbort(options.signal);
    assertTextEncoderDeclared(input.manifest, options, route);
    const buffer = (key: string): ArrayBuffer =>
      readAssetBuffer("WanPipeline", "weights / assets", input.assets, key);
    const open = await assetComponentOpener(
      "WanPipeline",
      input.assets,
      buffer,
      COMPONENT_KEYS[route],
    );
    const admitted = await WanPipeline.#admit(input.manifest, open, options);
    return await WanPipeline.#build(admitted, input.assets, open, options);
  }

  /**
   * この manifest と部品のグラフ宣言を Wan2.1 として実行できるかを見る（家族 admission — 取得面では
   * **重みの part を 1 バイトも取る前**に呼ばれる）。
   *
   * MUST: 家族の門はこの 1 本に集める（pipeline 名 / major・`pipelineConfig`・quant の `session`・
   * 共有 GPU の能力・計測の device・グラフ宣言 × ホストの取り決め）。後段へ散らすと、取得面では GB 級の
   * 重みを落とした**後**にしか落ちない（ADR 0070 決定 5）。
   * MUST: manifest の契約違反は **GPU を取りに行く前**に落とす（他の家族と同じ順序）。
   *
   * NOTE: RoPE の素表（`transformer` の容器の資産）はここで読む — 重み block と part を共有しない資産は
   * admission の席でも読める（`src/hub/components.ts`）。埋め込み資産・トークナイザ資産（manifest の
   * `assets`）はまだ届いていないので、中身の突合は {@link WanPipeline.#build}（ここで見るのは宣言の有無）。
   */
  static async #admit(
    manifest: Manifest,
    open: ComponentOpener,
    options: WanPipelineOptions,
  ): Promise<WanAdmission> {
    // 段の境目（容器を開いた後・重みの part を取る前）。
    await settleAbort(options.signal);
    const route = textEncoderRouteOf(options);
    const modelName = options.model ?? manifest.defaultModel;
    if (!Object.hasOwn(manifest.models, modelName)) {
      throw new Error(
        `WanPipeline: model '${modelName}' は manifest に無い` +
          `（利用可能: ${manifest.available.models.join(" / ")}）`,
      );
    }
    const entry: ModelEntry = manifest.models[modelName];
    const { name, major } = entry.pipeline;
    if (name !== WAN_PIPELINE_NAME) {
      throw new Error(
        `WanPipeline: manifest の pipeline が '${name}/${major}'` +
          `（'${WAN_PIPELINE_NAME}/${WAN_PIPELINE_MAJOR}' が必要）`,
      );
    }
    if (major !== WAN_PIPELINE_MAJOR) {
      // 「古い実装 × 新しいリポ」の沈黙劣化を止める唯一の門（ADR 0038 §6）。
      throw new Error(
        `WanPipeline: pipeline '${name}/${major}' の major に未対応` +
          `（この実装が読めるのは ${WAN_PIPELINE_NAME}/${WAN_PIPELINE_MAJOR}）`,
      );
    }
    const config = parseWanPipelineConfig(entry.pipelineConfig);

    const quantName = options.quant ?? entry.defaultQuant;
    if (!Object.hasOwn(entry.quants, quantName)) {
      throw new Error(
        `WanPipeline: quant '${quantName}' は manifest に無い` +
          `（利用可能: ${entry.available.quants.join(" / ")}）`,
      );
    }
    const quant = entry.quants[quantName];
    // `"gpu"` の経路はトークナイザ資産を読む — 宣言が無ければ umT5（約 5.3 GiB）を取る前に落とす
    // （中身は届いてから {@link WanPipeline.#build} が見る）。
    if (route === "gpu" && !Object.hasOwn(entry.assets, UMT5_TOKENIZER)) {
      throw new Error(
        `WanPipeline: manifest の assets に umT5 のトークナイザ資産 '${UMT5_TOKENIZER}' が無い` +
          '（textEncoder: "gpu" の経路が読む。事前計算の埋め込みだけで回すなら textEncoder: "precomputed"）',
      );
    }
    // 未対応の宣言は重みの part を取る前に落とす（全家族共通の 1 本）。
    const sessionOptions = resolveSessionOptions(
      WAN_SESSION_POLICY,
      quant.session,
      {},
      `WanPipeline: quant '${quantName}'`,
    );
    const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, sessionOptions);

    if (options.gpu?.gpuTimingEnabled === true) {
      throw new Error(
        "WanPipeline: gpuTiming が有効な device では VAE の段（1 タイル = 1 batch）を回せない" +
          "（runtime は計測の device で batch を開かない）— 計測なしの device を渡す",
      );
    }
    // MUST: 共有 GPU の能力不足（feature / device limit）はこの席で落とす — 重みを落とす前に判る
    // 唯一の家族門（後段の検査も同じ関数を呼ぶ）。
    if (options.gpu !== undefined) {
      assertGpuFeaturesGranted(gpuFeatures, options.gpu, `WanPipeline: quant '${quantName}'`);
      assertRequiredLimitsSatisfied(
        quant.requiredLimits,
        options.gpu.limits,
        `WanPipeline: quant '${quantName}'`,
      );
    }

    const transformer = open(TRANSFORMER);
    const layout = wanVaeChunkLayout(open(VAE_DECODER_FIRST), open(VAE_DECODER_NEXT));
    if (layout.latentChannels !== WAN_PATCH.channels) {
      throw new Error(
        `WanPipeline: VAE の潜在 ${layout.latentChannels} チャネルが DiT の ${WAN_PATCH.channels} と違う`,
      );
    }
    assertWanVaeTilesCoverAcceptedSizes(layout);
    const ropeBase = parseWanRopeBase(await readWholeAsset(transformer.asset(ROPE_BASE)));
    const dit = ditContract(transformer, ropeBase);
    const textEncoder = route === "gpu"
      ? { route, contract: umt5Contract(open(TEXT_ENCODER), dit) }
      : { route };
    return {
      config,
      quantName,
      quant,
      sessionOptions,
      gpuFeatures,
      layout,
      ropeBase,
      dit,
      textEncoder,
    };
  }

  /**
   * admission を通った材料 + 資産から組む（{@link WanPipeline.fromAssets} と `fromPretrained` が共有する
   * 1 本）。埋め込み資産を解析して DiT の文脈入力と突き合わせ、GPU を取る（共有 GPU なら取らない）。
   *
   * MUST: 資産の解析と突合は **GPU を取りに行く前**（壊れた資産の真因を GPU 無し環境の別の例外で
   * 消さない — 他の家族と同じ順序）。Session は 1 本も張らない（VRAM の MUST — モジュール doc）。
   */
  static async #build(
    admitted: WanAdmission,
    assets: WanAssets["assets"],
    open: ComponentOpener,
    options: WanPipelineOptions,
  ): Promise<WanPipeline> {
    const { config, quantName, sessionOptions, gpuFeatures, layout, ropeBase, dit } = admitted;
    // 段の境目（資産が届いた後・解析の前）。
    await settleAbort(options.signal);
    const textEmbeds = parseWanTextEmbeds(
      readAssetBuffer("WanPipeline", "weights / assets", assets, TEXT_EMBEDS),
    );
    assertEmbedsFitContext(dit, textEmbeds);
    let text: WanTextStage = { kind: "precomputed" };
    if (admitted.textEncoder.route === "gpu") {
      await settleAbort(options.signal);
      const encoder = new WanPromptEncoder(
        parseWanTokenizerAsset(
          readAssetJson("WanPipeline", "weights / assets", assets, UMT5_TOKENIZER),
          UMT5_TOKENIZER,
        ),
      );
      // トークナイザの上限はグラフの記号次元の上限・バケット表の生成器の上限と同じ値（決定 4）。
      // 違うと、上限の間のプロンプトがトークナイザを通ってからバケット表の生成で落ちる（大きい側）か、
      // 上流が受ける長さを黙って拒む（小さい側）。
      if (encoder.maxLength !== WAN_UMT5_MAX_LENGTH) {
        throw new Error(
          `WanPipeline: トークナイザ資産の maxLength ${encoder.maxLength} が umT5 の有効長の上限 ` +
            `${WAN_UMT5_MAX_LENGTH} と違う`,
        );
      }
      // 既定の negative はこの資産の門を通る MUST — 通らなければ資産の齟齬で、negativePrompt を省いた
      // 生成の入口で入力起因（`ModelInputError`）として落とすのは取り違え（呼び手の入力ではない）。
      try {
        encoder.encode(WAN_DEFAULT_NEGATIVE_PROMPT, "既定の negative");
      } catch (cause) {
        throw new Error(
          "WanPipeline: 既定の negative（公式の sample_neg_prompt）がトークナイザ資産の門を通らない",
          { cause },
        );
      }
      text = {
        kind: "gpu",
        encoder,
        component: open(TEXT_ENCODER),
        contract: admitted.textEncoder.contract,
      };
    }

    await settleAbort(options.signal);
    const gpu = options.gpu ?? await acquireGpu(toAcquireGpuOptions(gpuFeatures));
    const ownsGpu = options.gpu === undefined;
    try {
      // MUST: GPU 取得**後**の中断検査は try の中に置く — 外に出すと、内部で取った device を誰も
      // 解放できないまま抜ける。`acquireGpu` の await 明けはマイクロタスクなので、ここでもマクロタスクへ
      // 譲る（待機中に積まれた中断のタスクはまだ実行されていない — anima と同じ）。
      await settleAbort(options.signal);
      // 宣言された feature は device 作成時にしか要求できない（ADR 0028）— 自前で取った device は
      // ここが唯一の門（共有 GPU は {@link WanPipeline.#admit} が同じ 1 本で見ている）。
      assertGpuFeaturesGranted(gpuFeatures, gpu, `WanPipeline: quant '${quantName}'`);
      // 自前で取った device の limits もここで見る — `fromAssets` には取得前の事前判定の席が無く
      // （`fromPretrained` はアダプタ値で見ている）、見ないと宣言の不足が段ごとの Session の構築まで
      // 遅れる（VAE の段なら DiT の段を全部回した後）。共有 GPU は {@link WanPipeline.#admit} が見ている。
      if (ownsGpu) {
        assertRequiredLimitsSatisfied(
          admitted.quant.requiredLimits,
          gpu.limits,
          `WanPipeline: quant '${quantName}'`,
        );
      }
      return new WanPipeline({
        gpu,
        ownsGpu,
        config,
        sessionOptions,
        // 供給口は admission が見たものと**同じ 1 本**（`open` は開いた部品を引き当てるだけ）。
        transformer: open(TRANSFORMER),
        vaeFirst: open(VAE_DECODER_FIRST),
        vaeNext: open(VAE_DECODER_NEXT),
        layout,
        ropeBase,
        dit,
        textEmbeds,
        text,
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
   * テキスト埋め込み資産のプロンプトの一覧（資産のメタの並び — 埋め込みの値は持たない写し）。
   * 経路で意味が変わる:
   *
   * - `"precomputed"`: **受理集合そのもの**。`prompt` / `negativePrompt` には各行の `prompt` か
   *   `normalized` を渡す。
   * - `"gpu"`（既定）: 例示（固定プロンプトの名前と原文）。受理集合は任意の文字列で、各行の `prompt` を
   *   渡すと GPU の umT5 で符号化する（資産の埋め込みは使わない）。
   */
  get prompts(): readonly WanPrompt[] {
    return this.#state.textEmbeds.entries.map(({ name, role, prompt, normalized }) => ({
      name,
      role,
      prompt,
      normalized,
    }));
  }

  /**
   * プロンプトから動画を 1 本生成する。同じ要求（seed か latents・ノブ）なら同じ値が出る。
   * 並行に呼ばれた場合は待たされて順に走る（グラフの同時常駐を作らない — モジュール doc）。
   */
  async generate(request: WanGenerateRequest): Promise<GeneratedVideo> {
    if (this.#disposal !== undefined) throw new Error("WanPipeline: dispose 済みでは生成できない");
    return await this.#chain(() => this.#generate(request));
  }

  async #generate(request: WanGenerateRequest): Promise<GeneratedVideo> {
    const state = this.#state;
    const { dit } = state;
    const { onEvent, signal } = request;
    const emit = onEvent === undefined
      ? () => Promise.resolve()
      : async (event: WanGenerateEvent) => {
        await onEvent(event);
      };
    // 入口の門は経路ごと（どちらも GPU に触る前の純粋な検査）。プロンプトの文脈は経路ごとに作り、
    // DiT と VAE の段は同じ 1 本を通る。
    let knobs: WanGenerationKnobs;
    let contexts: WanContexts;
    if (state.text.kind === "gpu") {
      const plan = planWanGpuGeneration(request, state.text.encoder, state.layout, state.config);
      knobs = plan;
      // 段の境目（入口 → text）。直列化鎖の順番待ちの間に届いた中断もここで効く。
      await settleAbort(signal);
      contexts = await this.#encode(state.text, plan, emit, signal);
    } else {
      const plan = planWanGeneration(request, state.textEmbeds, state.layout, state.config);
      knobs = plan;
      contexts = {
        positive: padWanTextEmbedding(plan.positive, dit.contextRows, dit.contextWidth),
        negative: plan.negative === undefined
          ? undefined
          : padWanTextEmbedding(plan.negative, dit.contextRows, dit.contextWidth),
      };
    }
    // 段の境目（text → DiT）。
    await settleAbort(signal);
    const latents = await this.#denoise(knobs, contexts, emit, signal);
    // 段の境目（DiT → VAE）。
    await settleAbort(signal);
    const data = await this.#decode(knobs, latents, emit, signal);
    return { frames: knobs.frames, width: knobs.width, height: knobs.height, data };
  }

  /**
   * text の段（`"gpu"` の経路 — umT5 の Session を張り、positive → negative を 1 回ずつ回して畳む・`end` は
   * 畳んだ後）。出力 `[1, L, width]` を DiT の文脈の行数までゼロで詰めて返す（{@link padUmt5Context}）。
   *
   * MUST: DiT の段を張る前に畳む（ADR 0119 決定 11 — umT5 i8 5.30 GiB と DiT 段は B570 の天井に同居
   * できない）。畳む失敗で本体の失敗（run の失敗・中断・非有限の門）を上書きしない（DiT の段と同じ形）。
   * Session の実行オプションは `{}`（quant の `session` は DiT の Session だけが受ける — i8 の重みの実行は
   * 格納が決める・ADR 0119 決定 5）。
   *
   * MUST: 出力の有限性を見る（O(L·width)）。非有限の文脈を DiT へ渡すと、落ちるのは step 1 の潜在の門で、
   * 文言が DiT を指す（真因の段を取り違える）。
   */
  async #encode(
    text: Extract<WanTextStage, { readonly kind: "gpu" }>,
    plan: WanGenerationPlan<Int32Array<ArrayBuffer>>,
    emit: (event: WanGenerateEvent) => Promise<void>,
    signal: AbortSignal | undefined,
  ): Promise<WanContexts> {
    const state = this.#state;
    const { dit } = state;
    const observe = state.onRunDiagnostics;
    const prompts: readonly { readonly ids: Int32Array<ArrayBuffer>; readonly label: string }[] = [
      { ids: plan.positive, label: "prompt" },
      ...(plan.negative === undefined ? [] : [{ ids: plan.negative, label: "negativePrompt" }]),
    ];

    await emit({ kind: "stage", component: "text_encoder", at: "start" });
    const session = await text.component.createSession(state.gpu, {});
    const contexts: Float32Array<ArrayBuffer>[] = [];
    let failure: { readonly error: unknown } | undefined;
    try {
      for (const { ids, label } of prompts) {
        // 各 run の前（1 回の run は不可分 — 中断は次の run の前で効く）。
        await settleAbort(signal);
        const outputs = await session.run(
          umt5SessionInputs(ids, buildUmt5RelativePositionBuckets(ids.length)),
        );
        observe?.("text_encoder", session.diagnostics());
        if (!Object.hasOwn(outputs, text.contract.output)) {
          throw new Error(`WanPipeline: umT5 の出力 '${text.contract.output}' が無い`);
        }
        const context = padUmt5Context(
          outputs[text.contract.output],
          ids.length,
          dit.contextRows,
          dit.contextWidth,
        );
        const broken = firstNonFinite(context);
        if (broken !== -1) {
          throw new Error(
            `WanPipeline: umT5 の出力（${label}・${ids.length} トークン）の行 ` +
              `${Math.floor(broken / dit.contextWidth)}・列 ${broken % dit.contextWidth} が非有限` +
              `（${context[broken]}）— DiT の段へは渡さない`,
          );
        }
        contexts.push(context);
      }
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
        },
        () => session.dispose(),
      ]);
    }
    await emit({ kind: "stage", component: "text_encoder", at: "end" });
    return { positive: contexts[0], negative: contexts.at(1) };
  }

  /**
   * DiT の段（Session を張り、steps 回まわして畳む — `end` は畳んだ後）。
   *
   * MUST: 各 step の更新後の潜在の有限性を見る（O(N) のホスト走査 — 81 フレームで 1 step 約 210 万
   * 要素）。非有限の潜在を黙って次の step と VAE の段へ渡すと、VAE の後のクランプが ±Inf を ±1 に
   * 変えて検出できなくなる。
   */
  async #denoise(
    plan: WanGenerationKnobs,
    contexts: WanContexts,
    emit: (event: WanGenerateEvent) => Promise<void>,
    signal: AbortSignal | undefined,
  ): Promise<Float32Array<ArrayBuffer>> {
    const state = this.#state;
    const { dit } = state;
    const { latentShape, schedule } = plan;
    const grid = wanTokenGrid(latentShape, WAN_PATCH);
    const rope = wanRopeTables(state.ropeBase, grid);
    const tokenShape = [1, grid.count, wanTokenWidth(WAN_PATCH)];
    const ropeShape = [1, grid.count, 1, wanRopeWidth(state.ropeBase)];
    const contextShape = [1, dit.contextRows, dit.contextWidth];
    const { positive, negative } = contexts;
    const elements = latentShape.reduce((product, dim) => product * dim, 1);
    let current: Float32Array<ArrayBuffer> = plan.initial.kind === "latents"
      ? Float32Array.from(plan.initial.data)
      : new WanRandn(plan.initial.seed).normals(elements);
    const observe = state.onRunDiagnostics;

    await emit({ kind: "stage", component: "transformer", at: "start" });
    const session = await state.transformer.createSession(state.gpu, state.sessionOptions);
    let failure: { readonly error: unknown } | undefined;
    try {
      const predict = async (
        tokens: Float32Array<ArrayBuffer>,
        proj: Tensor,
        context: Float32Array<ArrayBuffer>,
      ): Promise<Float32Array<ArrayBuffer>> => {
        const outputs = await session.run({
          [DIT_TOKENS]: { dtype: "f32", shape: tokenShape, data: tokens },
          [DIT_TIMESTEPS_PROJ]: proj,
          [DIT_CONTEXT]: { dtype: "f32", shape: contextShape, data: context },
          [DIT_ROPE_COS]: { dtype: "f32", shape: ropeShape, data: rope.cos },
          [DIT_ROPE_SIN]: { dtype: "f32", shape: ropeShape, data: rope.sin },
        });
        observe?.("transformer", session.diagnostics());
        return unpatchifyTokens(asF32(outputs[dit.output], "DiT の出力"), latentShape, WAN_PATCH);
      };
      const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      for (let index = 0; index < plan.steps; index += 1) {
        // 各 step の前（step の 2 回の forward は不可分 — 中断は次の step の前で効く）。
        await settleAbort(signal);
        const timestep = schedule.timesteps[index];
        const proj: Tensor = {
          dtype: "f32",
          shape: [1, dit.projWidth],
          data: timestepsProj(timestep, dit.projWidth),
        };
        // CFG は uncond → cond の逐次 2 回（B = 1 — 決定 5）。同じ潜在なので patchify は 1 回。
        // 合成はホストで。
        const tokens = patchifyLatents(current, latentShape, WAN_PATCH);
        const uncond = negative === undefined ? undefined : await predict(tokens, proj, negative);
        const cond = await predict(tokens, proj, positive);
        const velocity = uncond === undefined
          ? cond
          : wanClassifierFreeGuidance(cond, uncond, plan.guidance);
        current = sampler.step(velocity, current);
        const broken = firstNonFinite(current);
        if (broken !== -1) {
          throw new Error(
            `WanPipeline: step ${index + 1}/${plan.steps} の更新後の潜在の要素 ${broken} が非有限` +
              `（${current[broken]}）— DiT の出力・CFG の合成・UniPC の更新のどこかが溢れた` +
              "（VAE の段へは渡さない）",
          );
        }
        const snapshot = current;
        await emit({
          kind: "denoise-step",
          step: index + 1,
          steps: plan.steps,
          timestep,
          copyLatents: () => ({ data: Float32Array.from(snapshot), shape: [...latentShape] }),
        });
      }
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      // MUST: 畳む失敗で本体の失敗（run の失敗・`onEvent` の throw・非有限の門）を上書きしない —
      // VAE の段と同じ形（`disposeSteps` の doc）。両方が落ちたら本体を先頭にした AggregateError。
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
        },
        () => session.dispose(),
      ]);
    }
    await emit({ kind: "stage", component: "transformer", at: "end" });
    return current;
  }

  /**
   * VAE の段（chunk グラフ 2 本の Session と常駐の cache を張り、タイル decode して畳む）。
   *
   * MUST: クランプの**前**に有限性を見る。上流と同じクランプ（{@link clampWanVaeFrames}）は ±Inf を
   * ±1 に変えるので、後では検出できない（クランプ自体は上流の写しなので変えない）。
   */
  async #decode(
    plan: WanGenerationKnobs,
    latents: Float32Array,
    emit: (event: WanGenerateEvent) => Promise<void>,
    signal: AbortSignal | undefined,
  ): Promise<Float32Array<ArrayBuffer>> {
    const state = this.#state;
    const { layout } = state;
    const tilePlan = plan.tiles;
    const tiles = wanVaeTileCount(tilePlan);
    const denormalized = denormalizeWanLatents(latents);
    const observe = state.onRunDiagnostics;

    await emit({ kind: "stage", component: "vae_decoder", at: "start" });
    let first: Session | undefined;
    let next: Session | undefined;
    let caches: WanVaeChunkCaches | undefined;
    let frames: Float32Array<ArrayBuffer>;
    let failure: { readonly error: unknown } | undefined;
    try {
      // VAE は quant の session を受けない（`{}` — anima の `withStage` と同じ取り決め。
      // `sessionOptions` は DiT の Session へ渡す実効設定 — {@link WanAdmission.sessionOptions}）。
      first = await state.vaeFirst.createSession(state.gpu, {});
      next = await state.vaeNext.createSession(state.gpu, {});
      caches = await WanVaeChunkCaches.create(state.gpu, layout);
      const sessions = { first, next };
      frames = await decodeWanVaeTiled(
        state.gpu,
        sessions,
        caches,
        tilePlan,
        denormalized,
        async (tile) => {
          observe?.("vae_decoder_first", sessions.first.diagnostics());
          observe?.("vae_decoder_next", sessions.next.diagnostics());
          await emit({ kind: "vae-tile", tile, tiles });
          // タイルの間（1 タイル = 1 batch で不可分 — 中断は次のタイルの前で効く）。
          await settleAbort(signal);
        },
      );
    } catch (error) {
      failure = { error };
      throw error;
    } finally {
      // MUST: Session → 常駐の順で畳む（段 4 / 5 の e2e と同じ）。畳む失敗で本体の失敗を上書きしない。
      const opened = { first, next, caches };
      await disposeSteps([
        () => {
          if (failure !== undefined) throw failure.error;
        },
        () => opened.first?.dispose(),
        () => opened.next?.dispose(),
        () => opened.caches?.dispose(),
      ]);
    }
    const expected = 3 * plan.frames * plan.height * plan.width;
    if (frames.length !== expected) {
      throw new Error(
        `WanPipeline: VAE の出力 ${frames.length} 要素が [3, ${plan.frames}, ${plan.height}, ${plan.width}] と違う`,
      );
    }
    const broken = firstNonFinite(frames);
    if (broken !== -1) {
      const plane = plan.height * plan.width;
      const pixel = broken % plane;
      const sheet = Math.floor(broken / plane);
      throw new Error(
        `WanPipeline: VAE の出力（クランプ前）の channel ${Math.floor(sheet / plan.frames)}・` +
          `フレーム ${sheet % plan.frames}・画素 (x=${pixel % plan.width}, ` +
          `y=${Math.floor(pixel / plan.width)}) が非有限（${frames[broken]}）`,
      );
    }
    clampWanVaeFrames(frames);
    await emit({ kind: "stage", component: "vae_decoder", at: "end" });
    return frames;
  }

  /**
   * 解放する。**内部で取得した GPU だけ**破棄する（`options.gpu` は呼び手の所有物）。
   * MUST: in-flight の生成の完了を待ってから破棄する（flush-before-destroy — 鎖に載せる）。
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(() => {
      if (this.#state.ownsGpu) this.#state.gpu.destroy();
    });
    return this.#disposal;
  }

  /** `await using` 対応 — {@link dispose} の別名。 */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}
