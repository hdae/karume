/**
 * Wan2.1 の配布形ミラー（`models/karume-wan2.1/`）を組むコマンド。配布形を読む e2e とホストのテストの SKIP 文言が同じ
 * コマンドを案内する（2.2 の {@link WAN_TI2V_ASSEMBLE_COMMAND} と同じ形 — `wan-ti2v-pipeline.ts`）。
 *
 * 副作用ゼロ: 文字列の定数だけを持つ。
 */

/**
 * 配布形を組むコマンド。umT5 の越境参照は仮の SHA（`--allow-placeholder-ref`）で焼く — ローカルのミラーを根にする e2e は
 * 取得元の `crossRepo` の mapping で `models/karume-umt5-xxl/` を渡すので、revision は読まない。
 */
export const WAN_ASSEMBLE_COMMAND =
  "cd tools/export-recipes && uv run python dist.py --pipeline wan " +
  "--ref-repo hdae/karume-umt5-xxl --ref-revision 0000000000000000000000000000000000000000 " +
  "--ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder --allow-placeholder-ref";
