/**
 * `@karume/models/depth-anything` — Depth Anything V2（画像 → 相対深度マップ）ファミリの
 * サブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link DepthAnythingPipeline.fromPretrained} /
 * {@link DepthAnythingPipeline.fromAssets}）/ 深度を出す（`estimate`）/ 解放する（`dispose`）。
 *
 * 画素の decode は利用者の責務で、resize / rescale / normalize は `estimate` の中で配布形の
 * 定数どおりに掛かる。返るのは**入力と同じ寸法の生の f32 相対深度**で、`[0, 1]` への正規化も
 * 着色も利用者側（可視化の決定であって推論の一部ではないため — `src/depth-anything/pipeline.ts`
 * のモジュール doc）。{@link Rgb8Image} と {@link DepthMap} は `estimate` の引数と返り値の型
 * なので、このサブパスからも名指しできるように出す。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { DepthAnythingPipeline } from "./src/depth-anything/pipeline.ts";
export type {
  DepthAnythingAssets,
  DepthAnythingFromPretrainedOptions,
  DepthAnythingPipelineOptions,
  DepthMap,
} from "./src/depth-anything/pipeline.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
