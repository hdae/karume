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
  Sbv2ProsodyDraft,
  Sbv2RunComponent,
} from "./src/sbv2/pipeline.ts";

/**
 * 公開配布リポの既定ソース（pin 済み commit SHA — ADR 0073）。`ref` 省略の
 * {@link Sbv2Pipeline.fromPretrained} が読む値そのもので、追従へ切り替える利用者が
 * `{ ...SBV2_DEFAULT_SOURCE, revision: "main" }` と綴れるように面へ出す。
 */
export { SBV2_DEFAULT_SOURCE } from "./src/sbv2/config.ts";

/**
 * 修正辞書エントリ（誤読・アクセントの差し替え）の綴りは `@hdae/yomi` が正本。**素通しで通す**
 * — ここで別名や変換層を作ると、辞書側の検証（読みのモーラ分割・アクセント型の範囲）と
 * 二重定義になる。
 */
export type { OverlayEntry } from "@hdae/yomi";
