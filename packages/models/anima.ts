/**
 * `@karume/models/anima` — Anima（テキスト → 画像）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link AnimaPipeline.fromPretrained} /
 * {@link AnimaPipeline.fromAssets}）/ 生成する（`generate`）/ 生成の途中経過を購読する
 * （`onEvent` — {@link AnimaGenerateEvent}）/ **途中 latent からプレビューを近似する**
 * （{@link denormalizeLatents} + {@link ANIMA_LATENTS_MEAN} / {@link ANIMA_LATENTS_STD}）/
 * 解放する（`dispose`）/ 解像度の綴りを扱う
 * （{@link parseResolution} / {@link formatResolution}）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

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

/**
 * 公開配布リポの既定ソース（pin 済み commit SHA — ADR 0073）。`ref` 省略の
 * {@link AnimaPipeline.fromPretrained} が読む値そのもので、追従へ切り替える利用者が
 * `{ ...ANIMA_DEFAULT_SOURCE, revision: "main" }` と綴れるように面へ出す。
 */
export { ANIMA_DEFAULT_SOURCE } from "./src/anima/config.ts";

/**
 * 素版（非 turbo）3 モデルが同居する配布リポの pin（ADR 0073）。既定ではないので、素版を組む
 * 利用者が `fromPretrained(ANIMA_BASE_SOURCE, { model })` と綴る（1 リポ = 3 モデルなので
 * リポ参照だけでは 1 本に決まらない）。pin の MUST は {@link ANIMA_DEFAULT_SOURCE} と同じ。
 */
export { ANIMA_BASE_SOURCE } from "./src/anima/config.ts";

/**
 * 途中 latent の逆正規化素材。`denoise-step` イベントの `copyLatents()`
 * （{@link AnimaLatentSnapshot}）が返すのは**正規化された**生 latent なので、そのまま色に
 * 起こしても意味のある絵にならない。
 *
 * VAE は DiT を解放した**後**にしかロードできない（VRAM の MUST）ため、パイプラインは毎 step の
 * VAE プレビューを提供しない — ステップ毎プレビューが要る消費側は、この 3 本で latent を
 * 逆正規化してから自前の近似（チャネルの線形射影など）に通す、というのがここを出す理由。
 * 逆正規化を消費側で書き直すと定数が二重に持たれる（VAE config にしかない値で、IR にも
 * 配布資産にも入っていない）。
 */
export { ANIMA_LATENTS_MEAN, ANIMA_LATENTS_STD, denormalizeLatents } from "./src/anima/latents.ts";

/**
 * 解像度の綴り（`1344x768` / 正方の略記 `512`）と受理集合。
 *
 * **受理集合の正本はこのパッケージ**（ADR 0038 §2 — manifest には書かない）なので、CLI や
 * UI が `WxH` を解釈する経路も同じ実装を通す。ここを出さないと綴りの契約が二重に持たれる。
 */
export { formatResolution, parseResolution } from "./src/anima/resolution.ts";
export type { ImageSize } from "./src/anima/resolution.ts";
