/**
 * `@karume/models/irodori` — Irodori（テキスト → 音声）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link IrodoriPipeline.fromPretrained} /
 * {@link IrodoriPipeline.fromAssets}）/ 生成する（`generate` / `generateLatent`）/
 * 生成の途中経過を購読する（`onEvent` — {@link IrodoriGenerateEvent}）/ 解放する（`dispose`）。
 *
 * 波形は f32 で出入りする（WAV に落とす `encodeWav` と、参照音声を読む `decodeWav` は barrel
 * 側 — どちらもファミリ非依存の共通処理）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

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

/**
 * 公開配布リポの既定ソース（pin 済み commit SHA — ADR 0073）。`ref` 省略の
 * {@link IrodoriPipeline.fromPretrained} が読む値そのもので、追従へ切り替える利用者が
 * `{ ...IRODORI_DEFAULT_SOURCE, revision: "main" }` と綴れるように面へ出す。
 */
export { IRODORI_DEFAULT_SOURCE } from "./src/irodori/config.ts";
