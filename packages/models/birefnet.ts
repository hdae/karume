/**
 * `@karume/models/birefnet` — BiRefNet 系（画像 → α マット / 背景抜き）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link BirefnetPipeline.fromPretrained} /
 * {@link BirefnetPipeline.fromAssets}）/ 切り抜く（`segment`）/ 解放する（`dispose`）。
 *
 * 画素の decode は利用者の責務で、resize / rescale / normalize は `segment` の中で配布形の
 * 定数どおりに掛かる。返るのは**入力と同じ寸法の 8bit α** 1 枚で、RGBA への合成は利用者側
 * （方針が用途で変わるため — `src/birefnet/pipeline.ts` のモジュール doc）。{@link Rgb8Image}
 * と {@link AlphaMatte} は `segment` の引数と返り値の型なので、このサブパスからも名指しできる
 * ように出す。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { BirefnetPipeline } from "./src/birefnet/pipeline.ts";
export type {
  AlphaMatte,
  BirefnetAssets,
  BirefnetFromPretrainedOptions,
  BirefnetPipelineOptions,
} from "./src/birefnet/pipeline.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
