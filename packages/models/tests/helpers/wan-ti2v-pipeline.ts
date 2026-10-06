/**
 * Wan2.2 TI2V-5B の配布形ミラー（`models/karume-wan2.2/` — ADR 0121 段 8）と席の名前。通しの e2e
 * （`e2e_wan_ti2v_pipeline_test.ts`）と DiT の e2e（`e2e_wan_ti2v_dit_test.ts`）が同じ置き場・同じ席名・同じ組み立ての
 * コマンドを読む。
 *
 * 配布形は `dist.py --pipeline wan-ti2v` が系列の容器（DiT は i8 の 1 系列・VAE 2 本は f16）と 2.1 の系列のテキスト資産 2 本
 * （ADR 0121 決定 9 — 同じ umT5 の埋め込みとトークナイザ）から組む。umT5 の容器は 2.1 と同じく別の配布リポ
 * （`karume-umt5-xxl`）への越境参照で、manifest の `text_encoder` の part が repo 名を持つ。
 *
 * 席の束（`session`）の正本は配布形の manifest の quant 席（ADR 0110 決定 1）で、ここには書き写さない — 実用席の束が要る
 * 呼び手は {@link readWanTi2vDistributionManifestText} で manifest を読み、席の `session` を引く。
 *
 * 副作用ゼロ: ここは URL と文字列の定数と、呼ばれたときだけ読む関数だけを持つ（モジュール評価で資産を読まない）。
 */

import { readTextIfPresent } from "./read-if-present.ts";

/** 参照席（ADR 0121 決定 2 — 重み i8・計算 f32・`session` 空）。 */
export const WAN_TI2V_REFERENCE_QUANT = "f16+dit8";
/** 実用席（ADR 0121 決定 2 — 2.1 の実用席と同じ名前と束）。 */
export const WAN_TI2V_PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";

/** 配布形ミラー（`dist.py --pipeline wan-ti2v` の既定の出力先）。 */
export const WAN_TI2V_DIST_ROOT = new URL("../../../../models/karume-wan2.2/", import.meta.url);
/** 配布形の Wan2.2 のモデル名（manifest の models のキー）。 */
export const WAN_TI2V_DIST_MODEL = "ti2v-5b";

/**
 * 配布形を組むコマンド。umT5 の越境参照は仮の SHA（`--allow-placeholder-ref`）で焼く — ローカルのミラーを根にする e2e は
 * 取得元の `crossRepo` の mapping で `models/karume-umt5-xxl/` を渡すので、revision は読まない。
 */
export const WAN_TI2V_ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline wan-ti2v " +
  "--ref-repo hdae/karume-umt5-xxl --ref-revision 0000000000000000000000000000000000000000 " +
  "--ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder --allow-placeholder-ref";

/**
 * 配布形の manifest のテキストを読む（`karume.json` が無ければ `undefined` — SKIP の判定の源）。NotFound 以外の失敗は投げる
 * （壊れた配布形を「無い」と読み替えて SKIP にしない）。parse はここでせず呼び手が使う所で行う — 壊れた manifest が赤にする
 * 範囲を、その manifest を読むケースだけに留めるため（DiT の e2e では実用席のケースだけ。2.1 の DiT の e2e と同じ置き方）。
 */
export const readWanTi2vDistributionManifestText = (): Promise<string | undefined> =>
  readTextIfPresent(new URL("karume.json", WAN_TI2V_DIST_ROOT));
