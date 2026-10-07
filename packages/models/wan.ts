/**
 * `@karume/models/wan` — Wan2.1 / 2.2（テキスト → 動画）ファミリのサブパス面（ADR 0118 決定 7・
 * ADR 0121 決定 10）。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link WanPipeline.fromPretrained} /
 * {@link WanPipeline.fromAssets}・Wan2.2 は {@link WanTi2vPipeline} の同名の 2 つ・テキストエンコーダの
 * 経路 `textEncoder`）/ 例示のプロンプトを引く
 * （`prompts`）/ 生成する（`generate`）/ 生成の途中経過を購読する（`onEvent` — {@link WanGenerateEvent}）/
 * 中断する（構築と生成の `signal`）/ フレームを画素にする（{@link wanFrameToRgba}）/ 解放する（`dispose`）。
 *
 * テキストエンコーダの経路は 2 つ（ADR 0119 決定 7）: 既定の `"gpu"` は umT5（i8・umT5 の配布リポへの
 * 越境参照）を GPU で回して任意のプロンプトを受け、`"precomputed"` は umT5 を取らずにテキスト埋め込み
 * 資産の固定プロンプトだけを受ける。
 *
 * NOTE: 取得元の対応表（`WAN_SOURCES`）はまだ無い — 配布形 `karume-wan2.1` / `karume-wan2.2` は HF へ
 * 未公開で、公開リポを持たない家族は表を持たない（ADR 0073 決定 1）。手元の配布形は `@karume/hub/deno` の
 * `denoDirectory` で `fromPretrained` へ渡し、`"gpu"` の経路では umT5 の配布形（`karume-umt5-xxl`）を
 * その `crossRepo` の mapping で渡す。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { WanPipeline } from "./src/wan/pipeline.ts";
/**
 * Wan2.2 TI2V 5B のテキスト → 動画（T2V）。構築・生成・中断・解放の面と公開型（要求・結果・イベント・
 * 構築オプション）は {@link WanPipeline} と同じで、受理集合（1280×704 / 704×1280・5〜121 フレーム）・
 * 潜在の形・fps は世代の値（ADR 0121 決定 10）。配布形は `karume-wan2.2`（ADR 0121 段 8 — recipe
 * `dist.py --pipeline wan-ti2v` が組む・HF には未公開）。
 */
export { WanTi2vPipeline } from "./src/wan/ti2v-pipeline.ts";
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
 * テキスト埋め込み資産のプロンプトの 1 行（資産のメタ — class の `prompts`〈`WanPipeline.prompts` /
 * `WanTi2vPipeline.prompts`〉の要素）と役割の語彙。
 * `"precomputed"` の経路では受理集合そのもの（集合の外の文字列は `ModelInputError`）、`"gpu"` の経路では
 * 例示なので、CLI / UI が選択肢を出すにはこの型が要る。
 */
export type { WanPrompt, WanPromptRole } from "./src/wan/text-embeds.ts";

/**
 * 生成した動画の 1 フレーム → RGBA 8bit（`encodePng` へそのまま渡せる）。uint8 化の規則
 * （`clamp(x/2 + 0.5, 0, 1)` → `round(·255)`）はパッケージが正本で、参照値（sha256）もこの規則の
 * バイト列で採る — 消費側に規則を書き直させない。
 */
export { wanFrameToRgba } from "./src/wan/frames.ts";

/**
 * 入力起因の失敗（渡した要求そのものが受理できない = 入力を直せば通る — 前処理・トークナイザが拒む
 * プロンプト〈文言が直し方を言う〉・集合の外のプロンプト・受理集合の外の寸法・値域外のノブ）。
 * **家族横断で 1 本**なので、複数の家族を同じホストに載せる側は
 * これだけで 400 と 500 を分けられる。内部不変条件の破れ・資産の齟齬・GPU 容量は素の `Error` の
 * まま飛ぶ。
 */
export { ModelInputError } from "./src/errors.ts";
