/**
 * `@karume/models/sbv2` — SBV2（テキスト → 音声）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link Sbv2Pipeline.fromPretrained} /
 * {@link Sbv2Pipeline.fromAssets}）/ 合成する（`generate`）/ 解放する（`dispose`）。
 *
 * スタイル名・話者名の受理集合は**配布形の `pipelineConfig`** が持つ（ADR 0038 §1）ので、
 * Anima の `parseResolution` に相当する「綴りの正本」はこのファミリには無い。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { Sbv2Pipeline } from "./src/sbv2/pipeline.ts";
export type {
  GeneratedAudio,
  Sbv2Assets,
  Sbv2FromPretrainedOptions,
  Sbv2GenerateRequest,
  Sbv2PipelineOptions,
} from "./src/sbv2/pipeline.ts";
