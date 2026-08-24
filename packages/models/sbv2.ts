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
 * 下書き（アクセント句 / モーラ）の語彙。`analyzeProsody` が返し、核を直したものを `generate` の
 * `prosody` へ戻す。**SBV2 が実際に読む欄だけ**を持つ（yomi の解析結果の部分集合 — 渡せるのに
 * 効かない欄を公開面に作らないため）。
 */
export type { Sbv2AccentPhrase, Sbv2Mora, Sbv2Prosody } from "./src/sbv2/text/prosody.ts";

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

/**
 * 修正辞書（誤読・アクセントの差し替え）の綴りは `@hdae/yomi` が正本。**素通しで通す** —
 * ここで別名や変換層を作ると、辞書側の検証（読みのモーラ分割・アクセント型の範囲）と
 * 二重定義になる。
 *
 * `OverlayDictionary` は**型だけ**を通す（構築するなら `@hdae/yomi` から値を import する）。
 * 常駐サーバーのように辞書が実行中に増減する側は、解決済みのものを持ち回って毎回の要求へ
 * 載せる（`Sbv2PipelineOptions.overlay` の MUST）。
 */
export type { OverlayDictionary, OverlayEntry } from "@hdae/yomi";
