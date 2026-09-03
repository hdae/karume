/**
 * `@karume/models/anima` — Anima（テキスト → 画像）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link AnimaPipeline.fromPretrained} /
 * {@link AnimaPipeline.fromAssets}）/ 生成する（`generate`）/ 生成の途中経過を購読する
 * （`onEvent` — {@link AnimaGenerateEvent}）/ **途中 latent からプレビューを近似する**
 * （{@link approximatePreview}）/ 解放する（`dispose`）/ 解像度の綴りを扱う
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
 * denoise の更新則の語彙（`AnimaGenerateRequest.sampler` が受ける値）。省略時は manifest の
 * `pipelineConfig.scheduler.type` が既定を決めるので、**指定は配布者の推奨からの明示的な離脱**
 * になる。型を出さないと消費側（CLI / UI）が選択肢を綴り直すことになるので面に出す。
 */
export type { AnimaSamplerType } from "./src/anima/config.ts";

/**
 * **このパッケージ版が検証した取得元**（公開配布リポ 1 つにつき 1 定数・pin 済み commit SHA —
 * ADR 0073）。リポの分割軸は「公式 / 追加学習」（ADR 0087）で 2 本:
 *
 * - `ANIMA_CURRENT` = `hdae/karume-anima`（公式モデル同居・既定 = Turbo）
 * - `ANIMA_EXTRA_CURRENT` = `hdae/karume-anima-extra`（第三者 fine-tune 同居・既定 = wai）
 *
 * どちらも既定以外は `{ model }` で選ぶ。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン** — `fromPretrained` の第 1 引数へ
 * そのまま渡す。再現性を自分で固定したい場合は、この定数ではなく自分の `{ repo, revision }` を
 * 書く（`fromPretrained` に既定は無い）。
 *
 * NOTE: 旧 `ANIMA_TURBO_CURRENT` は廃止（2026-09-01・breaking — `src/anima/config.ts` の
 * NOTE 参照）。
 */
export { ANIMA_CURRENT, ANIMA_EXTRA_CURRENT } from "./src/anima/config.ts";

/**
 * 途中 latent からの RGB プレビュー近似。`denoise-step` イベントの `copyLatents()`
 * （{@link AnimaLatentSnapshot}）が返す 2 欄を**そのまま**（逆正規化せずに）渡すと、latent
 * 解像度（元画像の 1/8）の RGBA が返る —
 * `const { data, shape } = event.copyLatents(); approximatePreview(data, shape)`。
 *
 * VAE は DiT を解放した**後**にしかロードできない（VRAM の MUST）ため、パイプラインは毎 step の
 * VAE プレビューを提供しない — step ごとの経過を絵で見せる手段はこの近似だけなので、面に出す。
 * 厳密な decode ではない（16ch → 3ch の線形射影）。係数の出所・較正空間の実測記録・限界は
 * 実装の doc が持つ。
 */
export { approximatePreview } from "./src/anima/preview.ts";

/**
 * 途中 latent の逆正規化素材（`const { mean, std } = animaLatents()` で取り、
 * `denormalizeLatents(latents, shape, mean, std)` へ渡す）。**プレビューには要らない** —
 * {@link approximatePreview} の係数は**正規化空間**で較正されている（2026-08-24 実測）ので、
 * 逆正規化した値を渡すと白飛びした別物になる。`copyLatents()` の返り値はそのまま渡す。
 *
 * ここを出すのは、VAE decode と同じ土俵で latent を扱いたい消費側（自前の decode・latent の
 * 解析・別の射影の較正）のため。この 2 本は VAE の config にしかなく、IR にも配布資産にも
 * 入っていないので、逆正規化を消費側で書き直すと定数が二重に持たれる。
 *
 * MUST: 定数の実体ではなく**アクセサ**を出す — {@link animaLatents} は呼び出しごとに独立した
 * 写しを返す。可変な TypedArray の実体を配ると、消費側の書き換えが `generate` の逆正規化へ
 * 黙って波及する（グローバル可変状態の禁止 — CLAUDE.md 横断不変条件）。
 * {@link denormalizeLatents} は mean / std を引数で受ける純関数なのでそのまま出す。
 */
export { animaLatents, denormalizeLatents } from "./src/anima/latents.ts";

/**
 * 解像度の綴り（`1344x768` / 正方の略記 `512`）と受理集合。
 *
 * **受理集合の正本はこのパッケージ**（ADR 0038 §2 — manifest には書かない）なので、CLI や
 * UI が `WxH` を解釈する経路も同じ実装を通す。ここを出さないと綴りの契約が二重に持たれる。
 */
export { formatResolution, parseResolution } from "./src/anima/resolution.ts";
export type { ImageSize } from "./src/anima/resolution.ts";
