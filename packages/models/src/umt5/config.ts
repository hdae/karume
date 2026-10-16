/**
 * umT5 の encoder（配布形 `karume-umt5-xxl`）の `pipeline` の契約名と `pipelineConfig` のスキーマ検証
 * （ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * `pipelineConfig` は空（`{}`）だけを受ける: 実行に要る数（入出力の形・幅・有効長の範囲・バケットの構成）は
 * グラフ宣言とホストの定数（`./relative-position.ts`）が持ち、配布形が宣言する欄は無い（recipe
 * `wan/umt5_distribution.py` の `UMT5_PIPELINE_CONFIG`）。欄が現れたら、読まないまま通さずに落とす（未知キーは
 * fail loudly — 綴りの違う宣言が黙って無視される形にしない）。
 *
 * 公開配布リポの対応表（{@link UMT5_SOURCES}）もここに置く（ADR 0092 — 家族の `config.ts` が表の置き場）。
 *
 * MUST: モジュール副作用ゼロ（CLAUDE.md）。
 */

import type { HubRepoRef } from "@karume/hub";
import { assertAllowedKeys, readRecord } from "../config/readers.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const UMT5_PIPELINE_NAME = "umt5-encoder";
export const UMT5_PIPELINE_MAJOR = 1;

/**
 * umT5 の**公開配布リポ対応表**（ADR 0092 — 家族 1 つにつき 1 表・**既定の席は無い**）。
 * 値は**このパッケージ版が検証した取得元**（pin 済み commit SHA — ADR 0073）。
 *
 * キーは HF リポ名の basename から `karume-` を落とした綴り（`"karume-" + key` がリポ名の
 * basename に戻る — この不変条件は `tests/sources_test.ts` の門が見る）。同一家族 = 1 リポ:
 *
 * - `"umt5-xxl"` = `hdae/karume-umt5-xxl`（モデル `"xxl"` の encoder とトークナイザ資産）
 *
 * NOTE: recipe の `wan.sources.UMT5_SOURCES`（`tools/export-recipes/wan/sources.py` — 書き手が読む本家
 * `google/umt5-xxl` の上流の pin の表）とは別物。こちらは焼いた配布形の公開リポを指す。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン**として渡す — 再現性を自分で
 * 固定したい場合は、この表ではなく自分の `{ repo, revision }` を書く（`fromPretrained` に
 * 既定は無い）。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。main 追従が要る利用者は
 * `{ ...UMT5_SOURCES["umt5-xxl"], revision: "main" }` を明示的に選ぶ。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 の維持義務を継承 — 手書き + 手順書ゲート）。
export const UMT5_SOURCES = {
  "umt5-xxl": {
    repo: "hdae/karume-umt5-xxl",
    revision: "c66a8db12e005ba37027f2afa5b15efee4a681a7",
  },
} as const satisfies Record<string, HubRepoRef>;

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査する（空のオブジェクトだけを受ける）。 */
export const assertUmt5PipelineConfig = (raw: unknown): void => {
  assertAllowedKeys(readRecord(raw, "pipelineConfig"), [], "pipelineConfig");
};
