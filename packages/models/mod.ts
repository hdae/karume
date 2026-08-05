/**
 * `@karume/models` — パイプライン群の barrel。ファミリ別サブパス（`./anima`）と両建て
 * （ADR 0037）。
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

/**
 * RGBA → PNG。**パイプライン非依存の共通処理**（画像生成モデルは総じて最後にこれを通す）
 * なので、ファミリのサブパスではなく barrel 直下に置く。
 */
export { encodePng } from "./src/image/png.ts";
