/**
 * `@karume/models/siglip2` — SigLIP2（画像 → 埋め込み）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link Siglip2Pipeline.fromPretrained} /
 * {@link Siglip2Pipeline.fromAssets}）/ 埋め込む（`embed`）/ 解放する（`dispose`）。
 *
 * 画素の decode は利用者の責務で、resize / rescale / normalize は `embed` の中で配布形の
 * 定数どおりに掛かる（`src/siglip2/pipeline.ts` のモジュール doc）。{@link Rgb8Image} だけは
 * `embed` の引数の型なので、このサブパスからも名指しできるように再輸出する。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { Siglip2Pipeline } from "./src/siglip2/pipeline.ts";
export type {
  Siglip2Assets,
  Siglip2FromPretrainedOptions,
  Siglip2PipelineOptions,
} from "./src/siglip2/pipeline.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
