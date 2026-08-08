/**
 * `@karume/models` — パイプライン群の barrel。ファミリ別サブパス（`./anima` / `./sbv2`）と
 * 両建て（ADR 0037）。
 *
 * ADR 0008 の流儀で**薄い面**にする — ここに並ぶのは「パイプラインを組んで生成する」「出た
 * 画像を書き出す」という利用者ストーリーだけで、内部モジュールの素通し再輸出はしない。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態・top-level 登録の禁止 —
 * CLAUDE.md）。barrel から 1 本だけ import したときに他ファミリが落ちるのは、この不変条件が
 * 成り立っているときだけ。
 */

export { AnimaPipeline } from "./src/anima/pipeline.ts";
export type {
  AnimaAssets,
  AnimaFromPretrainedOptions,
  AnimaGenerateRequest,
  AnimaPipelineOptions,
  GeneratedImage,
} from "./src/anima/pipeline.ts";
export type { ImageSize } from "./src/anima/resolution.ts";

export { Sbv2Pipeline } from "./src/sbv2/pipeline.ts";
export type {
  GeneratedAudio,
  Sbv2Assets,
  Sbv2FromPretrainedOptions,
  Sbv2GenerateRequest,
  Sbv2PipelineOptions,
} from "./src/sbv2/pipeline.ts";

/**
 * RGBA → PNG / f32 波形 → WAV。**パイプライン非依存の共通処理**（画像生成モデルは総じて
 * 最後に PNG を、音声生成モデルは WAV を通す）なので、ファミリのサブパスではなく barrel
 * 直下に置く。
 */
export { encodePng } from "./src/image/png.ts";
export { encodeWav } from "./src/audio/wav.ts";
