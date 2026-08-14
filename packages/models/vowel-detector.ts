/**
 * `@karume/models/vowel-detector` — 母音検出（日本語音声 → リップシンク用の `.lab`）ファミリの
 * サブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link VowelDetectorPipeline.fromPretrained} /
 * {@link VowelDetectorPipeline.fromAssets}）/ 検出する（`detect`）/ 解放する（`dispose`）。
 *
 * WAV の decode とリサンプルは利用者の責務（入口は 16kHz モノラルの `Float32Array` —
 * `src/vowel-detector/pipeline.ts` のモジュール doc）。`decodeWav` は barrel から出ている。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { VowelDetectorPipeline } from "./src/vowel-detector/pipeline.ts";
export type {
  VowelDetectorAssets,
  VowelDetectorFromPretrainedOptions,
  VowelDetectorPipelineOptions,
  VowelDetectorResult,
} from "./src/vowel-detector/pipeline.ts";
export type { LabSegment } from "./src/vowel-detector/postprocess.ts";
