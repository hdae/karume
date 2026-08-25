/**
 * `@karume/models/sbv2` — SBV2（テキスト → 音声）ファミリのサブパス面。
 *
 * ADR 0008: ここは**明示的に設計した薄い面**であり、内部モジュールの素通し再輸出はしない。
 * 面は利用者ストーリーに対応する — 組む（{@link Sbv2Pipeline.fromPretrained} /
 * {@link Sbv2Pipeline.fromAssets}）/ 発話にする（{@link toSbv2Utterance}）/ 合成する
 * （`generate`）/ 解放する（`dispose`）。
 *
 * **0.6.0 の再裁定（breaking）**: テキスト解析（辞書・修正辞書）は呼び手の責務になり、
 * `generate` は解析済みの {@link Sbv2Utterance} を第 1 引数に受ける。ADR 0072 の注入席
 * （`text` / `overlay` / `prosody` / `givenTone` / `analyzeProsody`）とその語彙はシム無しで
 * 撤去した — 解析器を持たないぶん、入口は「渡された構造が自己整合していること」だけを見る。
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
  Sbv2GenerateOptions,
  Sbv2PipelineOptions,
  Sbv2RunComponent,
} from "./src/sbv2/pipeline.ts";

/**
 * 句構造の解析結果 → 発話の変換（純関数・GPU 不要）。核をモーラごとの 2 値トーンへ展開し、
 * 句の記号を末尾モーラへ寄せる。**アクセント句と核で読みを返す解析器の出力**は
 * {@link Sbv2Phrases} を構造的に満たすので、そのまま渡せる。
 */
export { toSbv2Utterance } from "./src/sbv2/text/utterance.ts";

/**
 * 発話の語彙。`moras` が編集面（`tone` を直すとアクセントが動く）で、`words` は読み取り専用。
 * **SBV2 が実際に読む欄だけ**を持つ（渡せるのに効かない欄を公開面に作らないため）。
 */
export type { Sbv2Mora, Sbv2Phrases, Sbv2Utterance, Sbv2Word } from "./src/sbv2/text/utterance.ts";

/**
 * 入力起因の失敗（呼び手が渡した要求が受理できない）。内部不変条件の破れは素の `Error` のまま
 * 飛ぶので、HTTP サーバーなら `instanceof` で 400 と 500 を分けられる。
 */
export { Sbv2InputError } from "./src/sbv2/errors.ts";

/**
 * **このパッケージ版が検証した取得元**（`hdae/karume-sbv2-jvnv` の pin 済み commit SHA —
 * ADR 0073）。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン** — {@link Sbv2Pipeline.fromPretrained}
 * の第 1 引数へそのまま渡す。再現性を自分で固定したい場合は、この定数ではなく自分の
 * `{ repo, revision }` を書く（`fromPretrained` に既定は無い）。
 */
export { SBV2_JVNV_CURRENT } from "./src/sbv2/config.ts";
