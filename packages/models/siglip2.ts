/**
 * `@karume/models/siglip2` — SigLIP2（画像 → 埋め込み）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link Siglip2Pipeline.fromPretrained} /
 * {@link Siglip2Pipeline.fromAssets}）/ 取得元を選ぶ（{@link SIGLIP2_SOURCES}）/ 埋め込む
 * （`embed`）/ 解放する（`dispose`）。
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
/**
 * **このパッケージ版が検証した取得元の対応表**（家族 1 つにつき 1 表・キーは HF リポ名から
 * `karume-` を落とした綴り・値は pin 済み commit SHA — ADR 0073 / 0092）。同一家族 = 1 リポ
 * なので 1 エントリで、寸法だけが違う 2 モデルが同居する:
 *
 * - `SIGLIP2_SOURCES["siglip2"]` = `hdae/karume-siglip2`（base 224 / so400m 384・既定 = base）
 *
 * 既定以外は `fromPretrained(SIGLIP2_SOURCES["siglip2"], { model: "so400m" })` と綴る
 * （{@link Siglip2PipelineOptions}）。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン** — {@link Siglip2Pipeline.fromPretrained}
 * の第 1 引数へそのまま渡す。再現性を自分で固定したい場合は、この表ではなく自分の
 * `{ repo, revision }` を書く（`fromPretrained` に既定は無い）。
 */
export { SIGLIP2_SOURCES } from "./src/siglip2/config.ts";
export type { Rgb8Image } from "./src/image/preprocess.ts";
