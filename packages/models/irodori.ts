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
 * **このパッケージ版が検証した取得元**（公開配布リポ 1 つにつき 1 定数・pin 済み commit SHA —
 * ADR 0073）。上流の v4 / v4.1 は別リポなので 2 本
 * （`hdae/karume-irodori-v4-small` / `hdae/karume-irodori-v4.1-small`）— v4.1 は duration
 * predictor だけを再学習した後続版で、旧版の pin も温存する。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン** — {@link IrodoriPipeline.fromPretrained}
 * の第 1 引数へそのまま渡す。再現性を自分で固定したい場合は、この定数ではなく自分の
 * `{ repo, revision }` を書く（`fromPretrained` に既定は無い）。
 */
export { IRODORI_V4_1_SMALL_CURRENT, IRODORI_V4_SMALL_CURRENT } from "./src/irodori/config.ts";
