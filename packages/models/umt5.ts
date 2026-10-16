/**
 * `@karume/models/umt5` — umT5-XXL（テキスト → encoder の最終隠れ状態）のサブパス面（配布形 `karume-umt5-xxl` —
 * pipeline `umt5-encoder/1`）。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。面は利用者ストーリーに
 * 対応する — 組む（{@link Umt5Encoder.fromPretrained} / {@link Umt5Encoder.fromAssets}）/ 取得元を選ぶ
 * （{@link UMT5_SOURCES}）/ トークナイザだけを通す
 * （`tokenize`）/ 符号化する（`encode` — {@link Umt5Encoding}）/ 中断する（`encode` の `signal`）/ 解放する
 * （`dispose`）。
 *
 * 入力の規則（前処理なし・語彙外の文字・本文中の特殊トークン・空白の直後の `▁`・U+0020 以外の空白と末尾の
 * 空白・512 トークン超・空文字を拒む）は `src/umt5/tokenizer.ts` のモジュール doc が正本。Wan の
 * `WanPipeline` / `WanTi2vPipeline` は同じトークナイザの前に上流の前処理（`prompt_clean`）を通す。
 *
 * 手元の配布形は `@karume/hub/deno` の `denoDirectory` で `fromPretrained` へ渡す。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 * barrel（`mod.ts`）経由の tree-shaking はこの不変条件の上にだけ成立する。
 */

export { Umt5Encoder } from "./src/umt5/encoder.ts";
export type {
  Umt5EncodeOptions,
  Umt5EncoderAssets,
  Umt5EncoderFromPretrainedOptions,
  Umt5EncoderOptions,
  Umt5Encoding,
} from "./src/umt5/encoder.ts";
/**
 * **このパッケージ版が検証した取得元の対応表**（家族 1 つにつき 1 表・キーは HF リポ名から
 * `karume-` を落とした綴り・値は pin 済み commit SHA — ADR 0073 / 0092）。同一家族 = 1 リポ
 * なので 1 エントリ:
 *
 * - `UMT5_SOURCES["umt5-xxl"]` = `hdae/karume-umt5-xxl`（モデル `"xxl"` の encoder とトークナイザ資産）
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン** — {@link Umt5Encoder.fromPretrained}
 * の第 1 引数へそのまま渡す。再現性を自分で固定したい場合は、この表ではなく自分の
 * `{ repo, revision }` を書く（`fromPretrained` に既定は無い）。
 */
export { UMT5_SOURCES } from "./src/umt5/config.ts";

/**
 * 入力起因の失敗（渡した要求そのものが受理できない = 入力を直せば通る — トークナイザが拒む文字列〈文言が
 * 直し方を言う〉）。**家族横断で 1 本**なので、複数の家族を同じホストに載せる側はこれだけで 400 と 500 を
 * 分けられる。内部不変条件の破れ・資産の齟齬・GPU 容量は素の `Error` のまま飛ぶ。
 */
export { ModelInputError } from "./src/errors.ts";
