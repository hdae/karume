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

export { Sbv2Pipeline } from "./src/sbv2/pipeline.ts";
export type {
  GeneratedAudio,
  Sbv2Assets,
  Sbv2FromPretrainedOptions,
  Sbv2GenerateRequest,
  Sbv2PipelineOptions,
  Sbv2RunComponent,
} from "./src/sbv2/pipeline.ts";

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
