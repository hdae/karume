/**
 * umT5 の encoder（配布形 `karume-umt5-xxl`）の `pipeline` の契約名と `pipelineConfig` のスキーマ検証
 * （ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * `pipelineConfig` は空（`{}`）だけを受ける: 実行に要る数（入出力の形・幅・有効長の範囲・バケットの構成）は
 * グラフ宣言とホストの定数（`./relative-position.ts`）が持ち、配布形が宣言する欄は無い（recipe
 * `wan/umt5_distribution.py` の `UMT5_PIPELINE_CONFIG`）。欄が現れたら、読まないまま通さずに落とす（未知キーは
 * fail loudly — 綴りの違う宣言が黙って無視される形にしない）。
 *
 * NOTE: 公開配布リポの対応表（`UMT5_SOURCES`）はまだ無い — 配布形 `karume-umt5-xxl` は HF へ未公開で、公開リポを
 * 持たない家族は表を持たない（ADR 0073 決定 1）。HF 公開の pin 焼き込みの回にここへ足す
 * （docs/release-runbook.md）。
 *
 * MUST: モジュール副作用ゼロ（CLAUDE.md）。
 */

import { assertAllowedKeys, readRecord } from "../config/readers.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const UMT5_PIPELINE_NAME = "umt5-encoder";
export const UMT5_PIPELINE_MAJOR = 1;

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査する（空のオブジェクトだけを受ける）。 */
export const assertUmt5PipelineConfig = (raw: unknown): void => {
  assertAllowedKeys(readRecord(raw, "pipelineConfig"), [], "pipelineConfig");
};
