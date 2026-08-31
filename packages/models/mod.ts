/**
 * `@karume/models` — パイプライン群の barrel。ファミリ別サブパス（`./anima` / `./birefnet` /
 * `./depth-anything` / `./gemma` / `./irodori` / `./sbv2` / `./siglip2` / `./vowel-detector`）と
 * 両建て（ADR 0037）。
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
/** `AnimaGenerateRequest.sampler` の語彙（`./anima` を参照）。 */
export type { AnimaSamplerType } from "./src/anima/config.ts";
/**
 * 解像度の綴り（`1344x768` / 正方の略記 `512`）と受理集合。**受理集合の正本はこのパッケージ**
 * （ADR 0038 §2）なので、{@link ImageSize} だけ出して生成器を出さないと、CLI / UI が `WxH` の
 * 綴り契約を書き直すことになる（`./anima` を参照）。
 */
export { formatResolution, parseResolution } from "./src/anima/resolution.ts";
export type { ImageSize } from "./src/anima/resolution.ts";
/**
 * 途中 latent → RGB の線形近似。`denoise-step` の `copyLatents()`
 * （{@link AnimaLatentSnapshot}）が返す 2 欄をそのまま渡す。VAE は DiT を解放した**後**にしか
 * ロードできない（VRAM の MUST）ので、step ごとの経過を絵で見せる手段はこれだけ — snapshot 型を
 * 出しておいて近似を出さないと、購読側は受け取った latent を使えない（`./anima` を参照）。
 */
export { approximatePreview } from "./src/anima/preview.ts";
/**
 * 途中 latent の逆正規化素材（`const { mean, std } = animaLatents()` → `denormalizeLatents`）。
 * **プレビューには要らない** — VAE decode と同じ土俵で latent を扱いたい消費側のための面で、
 * 出さないと 2 本の定数が消費側に二重持ちされる（`./anima` を参照）。
 */
export { animaLatents, denormalizeLatents } from "./src/anima/latents.ts";
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
  Sbv2GenerateOptions,
  Sbv2PipelineOptions,
  Sbv2RunComponent,
} from "./src/sbv2/pipeline.ts";
/**
 * 句構造の解析結果 → 発話の変換（`./sbv2` を参照）。0.6.0 でテキスト解析は呼び手の責務に
 * なり、`generate` は発話を第 1 引数に受ける（ADR 0072 の注入席はシム無しで撤去）。
 */
export { toSbv2Utterance } from "./src/sbv2/text/utterance.ts";
/** 発話（モーラ列 + 語アライメント）の語彙 — SBV2 が読む欄だけを持つ（`./sbv2` を参照）。 */
export type { Sbv2Mora, Sbv2Phrases, Sbv2Utterance, Sbv2Word } from "./src/sbv2/text/utterance.ts";
/** 入力起因の失敗（内部不変条件の破れは素の `Error` のまま — `./sbv2` を参照）。 */
export { Sbv2InputError } from "./src/sbv2/errors.ts";
/** このパッケージ版が検証した取得元（`./sbv2` を参照 — 追従したい場合のオプトイン）。 */
export { SBV2_JVNV_CURRENT } from "./src/sbv2/config.ts";

export { Siglip2Pipeline } from "./src/siglip2/pipeline.ts";
export type {
  Siglip2Assets,
  Siglip2FromPretrainedOptions,
  Siglip2PipelineOptions,
} from "./src/siglip2/pipeline.ts";

export { Gemma4Pipeline } from "./src/gemma/pipeline.ts";
export type {
  Gemma4Assets,
  Gemma4ChatOptions,
  Gemma4ChatStream,
  Gemma4FromPretrainedOptions,
  Gemma4PipelineOptions,
} from "./src/gemma/pipeline.ts";
export type { Gemma4PipelineConfig } from "./src/gemma/config.ts";
/**
 * 会話 → token id 列（`<bos>` 込み — 素の会話だけを受け、tools / thinking は fail loudly。
 * `./gemma` を参照）。`chat` は内部でこれを通すので、要るのは低レベル面（`sequence`）を
 * 自分で回すときだけ。
 */
export { gemma4ChatPrompt } from "./src/gemma/text/chat.ts";
export type { Gemma4ChatMessage, Gemma4ChatRole } from "./src/gemma/text/chat.ts";
/** 低レベル面で受けた token id を文字列へ戻す口（`./gemma` を参照）。 */
export type { GemmaDecodeOptions, GemmaTokenizer } from "./src/gemma/text/tokenizer.ts";
export type { StreamingDetokenizer } from "./src/text/detokenizer.ts";
/**
 * 生成の語彙（ADR 0083）。`Gemma4Pipeline.sequence()` を自分で回すときに要る型と、「会話が
 * 入り切らない」を他の失敗と読み分けるための例外（決定 10 — `./gemma` を参照）。
 */
export { GenerationCapacityError } from "./src/generation/sequence.ts";
export type {
  GenerationEvent,
  GenerationRequest,
  GenerationSequence,
  GenerationStop,
  GenerationStream,
} from "./src/generation/sequence.ts";
export type { GenerationProgram } from "./src/generation/program.ts";
export type { SamplerSpec } from "./src/generation/sampler.ts";

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
 * NOTE: 固定長 greedy 生成ループ（`generateGreedy` / `GreedySpec`）はここから**外した**
 * （ADR 0083 決定 9 — 破壊的変更。消費側の doc は limitations）。parity 検収用の内部ヘルパ
 * として `src/generation/greedy.ts` に残っており、生成の公開面は生成 API 波の sequence が持つ。
 */
