/**
 * `@karume/models/irodori` — Irodori（テキスト → 音声）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link IrodoriPipeline.fromPretrained} /
 * {@link IrodoriPipeline.fromAssets}）/ 生成する（`generate` / `generateLatent`）/
 * 解放する（`dispose`）。
 *
 * 波形は f32 で返る（WAV に落とすのは barrel の `encodeWav` — ファミリ非依存の共通処理）。
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
  IrodoriGenerateRequest,
  IrodoriPipelineOptions,
  IrodoriRunComponent,
  IrodoriSpeakerInput,
} from "./src/irodori/pipeline.ts";
