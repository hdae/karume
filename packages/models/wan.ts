/**
 * `@karume/models/wan` — Wan2.1（テキスト → 動画）ファミリのサブパス面（ADR 0118 決定 7）。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link WanPipeline.fromPretrained} /
 * {@link WanPipeline.fromAssets}）/ 受理するプロンプトを引く（`prompts`）/ 生成する（`generate`）/
 * 生成の途中経過を購読する（`onEvent` — {@link WanGenerateEvent}）/ フレームを画素にする
 * （{@link wanFrameToRgba}）/ 解放する（`dispose`）。
 *
 * NOTE: 取得元の対応表（`WAN_SOURCES`）はまだ無い — 配布形 `karume-wan2.1` は HF へ未公開で、公開
 * リポを持たない家族は表を持たない（ADR 0073 決定 1）。手元の配布形は `@karume/hub/deno` の
 * `denoDirectory` で `fromPretrained` へ渡す。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { WanPipeline } from "./src/wan/pipeline.ts";
export type {
  GeneratedVideo,
  WanAssets,
  WanFromPretrainedOptions,
  WanGenerateEvent,
  WanGenerateRequest,
  WanLatentSnapshot,
  WanPipelineOptions,
  WanRunComponent,
} from "./src/wan/pipeline.ts";

/**
 * 受理集合の 1 行（テキスト埋め込み資産のメタ — `WanPipeline.prompts` の要素）と役割の語彙。
 * 第 1 段（事前計算した埋め込み）は集合の外の文字列を `ModelInputError` で拒むので、CLI / UI が
 * 選択肢を出すにはこの型が要る。
 */
export type { WanPrompt, WanPromptRole } from "./src/wan/text-embeds.ts";

/**
 * 生成した動画の 1 フレーム → RGBA 8bit（`encodePng` へそのまま渡せる）。uint8 化の規則
 * （`clamp(x/2 + 0.5, 0, 1)` → `round(·255)`）はパッケージが正本で、参照値（sha256）もこの規則の
 * バイト列で採る — 消費側に規則を書き直させない。
 */
export { wanFrameToRgba } from "./src/wan/frames.ts";

/**
 * 入力起因の失敗（渡した要求そのものが受理できない = 入力を直せば通る — 集合の外のプロンプト・
 * 受理集合の外の寸法・値域外のノブ）。**家族横断で 1 本**なので、複数の家族を同じホストに載せる側は
 * これだけで 400 と 500 を分けられる。内部不変条件の破れ・資産の齟齬・GPU 容量は素の `Error` の
 * まま飛ぶ。
 */
export { ModelInputError } from "./src/errors.ts";
