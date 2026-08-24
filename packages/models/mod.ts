/**
 * `@karume/models` — パイプライン群の barrel。ファミリ別サブパス（`./anima` / `./birefnet` /
 * `./depth-anything` / `./irodori` / `./sbv2` / `./siglip2` / `./vowel-detector`）と両建て
 * （ADR 0037）。
 *
 * ADR 0008 の流儀で**薄い面**にする — ここに並ぶのは「パイプラインを組んで生成する」「出た
 * 画像を書き出す」という利用者ストーリーだけで、内部モジュールの素通し再輸出はしない。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態・top-level 登録の禁止 —
 * CLAUDE.md）。barrel から 1 本だけ import したときに他ファミリが落ちるのは、この不変条件が
 * 成り立っているときだけ。
 *
 * NOTE: `fromPretrained` が読む配布 manifest の `format` は版ごとに固定で、**旧版は読まない**。
 * 「パッケージ版 ⇔ `format`」の対応表は `@karume/hub` のモジュール doc（配布形を作り直す
 * 段取りを事前に読むための表）。
 */

export { BirefnetPipeline } from "./src/birefnet/pipeline.ts";
export type {
  AlphaMatte,
  BirefnetAssets,
  BirefnetFromPretrainedOptions,
  BirefnetPipelineOptions,
} from "./src/birefnet/pipeline.ts";

export { DepthAnythingPipeline } from "./src/depth-anything/pipeline.ts";
export type {
  DepthAnythingAssets,
  DepthAnythingFromPretrainedOptions,
  DepthAnythingPipelineOptions,
  DepthMap,
} from "./src/depth-anything/pipeline.ts";

export { AnimaPipeline } from "./src/anima/pipeline.ts";
export type {
  AnimaAssets,
  AnimaFromPretrainedOptions,
  AnimaGenerateEvent,
  AnimaGenerateRequest,
  AnimaLatentSnapshot,
  AnimaPipelineOptions,
  AnimaRunComponent,
  GeneratedImage,
} from "./src/anima/pipeline.ts";
export type { ImageSize } from "./src/anima/resolution.ts";
/**
 * 途中 latent → RGB の線形近似。`denoise-step` の `copyLatents()`
 * （{@link AnimaLatentSnapshot}）が返す 2 欄をそのまま渡す。VAE は DiT を解放した**後**にしか
 * ロードできない（VRAM の MUST）ので、step ごとの経過を絵で見せる手段はこれだけ — snapshot 型を
 * 出しておいて近似を出さないと、購読側は受け取った latent を使えない（`./anima` を参照）。
 */
export { approximatePreview } from "./src/anima/preview.ts";
/** このパッケージ版が検証した取得元（`./anima` を参照 — 追従したい場合のオプトイン）。 */
export { ANIMA_CURRENT, ANIMA_TURBO_CURRENT } from "./src/anima/config.ts";

export { IrodoriPipeline } from "./src/irodori/pipeline.ts";
export type {
  GeneratedLatent,
  IrodoriAssets,
  IrodoriFromPretrainedOptions,
  IrodoriGeneratedAudio,
  IrodoriGenerateEvent,
  IrodoriGenerateRequest,
  IrodoriLatentSnapshot,
  IrodoriPipelineOptions,
  IrodoriRunComponent,
  IrodoriSpeakerInput,
} from "./src/irodori/pipeline.ts";
/** このパッケージ版が検証した取得元（`./irodori` を参照 — 追従したい場合のオプトイン）。 */
export { IRODORI_V4_SMALL_CURRENT } from "./src/irodori/config.ts";

export { Sbv2Pipeline } from "./src/sbv2/pipeline.ts";
export type {
  GeneratedAudio,
  Sbv2Assets,
  Sbv2FromPretrainedOptions,
  Sbv2GenerateRequest,
  Sbv2PipelineOptions,
  Sbv2ProsodyDraft,
  Sbv2RunComponent,
} from "./src/sbv2/pipeline.ts";
/** 下書き（アクセント句 / モーラ）の語彙 — SBV2 が読む欄だけを持つ（`./sbv2` を参照）。 */
export type { Sbv2AccentPhrase, Sbv2Mora, Sbv2Prosody } from "./src/sbv2/text/prosody.ts";
/** 入力起因の失敗（内部不変条件の破れは素の `Error` のまま — `./sbv2` を参照）。 */
export { Sbv2InputError } from "./src/sbv2/errors.ts";
/** このパッケージ版が検証した取得元（`./sbv2` を参照 — 追従したい場合のオプトイン）。 */
export { SBV2_JVNV_CURRENT } from "./src/sbv2/config.ts";
/** 修正辞書の綴りは `@hdae/yomi` が正本（`./sbv2` と同じく素通し — 変換層は作らない）。 */
export type { OverlayDictionary, OverlayEntry } from "@hdae/yomi";

export { Siglip2Pipeline } from "./src/siglip2/pipeline.ts";
export type {
  Siglip2Assets,
  Siglip2FromPretrainedOptions,
  Siglip2PipelineOptions,
} from "./src/siglip2/pipeline.ts";

export { VowelDetectorPipeline } from "./src/vowel-detector/pipeline.ts";
export type {
  VowelDetectorAssets,
  VowelDetectorFromPretrainedOptions,
  VowelDetectorPipelineOptions,
  VowelDetectorResult,
} from "./src/vowel-detector/pipeline.ts";
export type { LabSegment } from "./src/vowel-detector/postprocess.ts";

/**
 * RGBA → PNG / f32 波形 ↔ WAV / RGB8 → モデル入力。**パイプライン非依存の共通処理**
 * （画像生成モデルは総じて最後に PNG を、音声生成モデルは WAV を通し、声質の参照には WAV を、
 * 画像を入力に取るモデルは resize → rescale → normalize を通る）なので、ファミリの
 * サブパスではなく barrel 直下に置く。
 */
export { encodePng } from "./src/image/png.ts";
export { normalizeToNchw, resizeRgb8 } from "./src/image/preprocess.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
export { decodeWav, encodeWav } from "./src/audio/wav.ts";
export type { DecodedWav } from "./src/audio/wav.ts";

/**
 * 固定長 greedy 生成ループ。**パイプライン非依存の共通処理**（autoregressive な言語モデルは
 * 総じて「固定長 chunk の prefill → 1 token ずつ decode」を通る — ADR 0066 決定 4）なので、
 * `encodePng` / `decodeWav` と同じくファミリのサブパスではなく barrel 直下に置く。
 */
export { generateGreedy } from "./src/generation/greedy.ts";
export type { GreedySpec } from "./src/generation/greedy.ts";
