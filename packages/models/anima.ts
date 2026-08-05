/**
 * `@karume/models/anima` — Anima（テキスト → 画像）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link AnimaPipeline.fromPretrained} /
 * {@link AnimaPipeline.fromAssets}）/ 生成する（`generate`）/ 解放する（`dispose`）/
 * 解像度の綴りを扱う（{@link parseResolution} / {@link formatResolution}）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { AnimaPipeline } from "./src/anima/pipeline.ts";
export type {
  AnimaAssets,
  AnimaFromPretrainedOptions,
  AnimaGenerateRequest,
  AnimaPipelineOptions,
  GeneratedImage,
} from "./src/anima/pipeline.ts";

/**
 * 解像度の綴り（`1344x768` / 正方の略記 `512`）と受理集合。
 *
 * **受理集合の正本はこのパッケージ**（ADR 0038 §2 — manifest には書かない）なので、CLI や
 * UI が `WxH` を解釈する経路も同じ実装を通す。ここを出さないと綴りの契約が二重に持たれる。
 */
export { formatResolution, parseResolution } from "./src/anima/resolution.ts";
export type { ImageSize } from "./src/anima/resolution.ts";
