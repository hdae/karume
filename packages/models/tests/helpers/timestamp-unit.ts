/**
 * timestamp-query の値の 1 単位を ns へ直す表（環境キー別 — Wan の DiT / umT5 の e2e が共有する）。
 *
 * Deno は wgpu の raw tick を換算せずに返す（docs/known-issues.md「Intel Arc B570」節 — B570 で 1 tick = 52.0833 ns）。
 * 表に無い環境の換算は推測しない: 門として時間を見る呼び手は {@link timestampUnitNs} で fail loudly にし、記録だけの
 * 呼び手は {@link lookupTimestampUnitNs} の undefined を見て換算を飛ばす。
 *
 * 環境キーは引数で受ける（このモジュールは何も import しない — 副作用ゼロ）。
 */

/** timestamp の 1 単位の ns（環境キー → ns）。行を足すのは実測で period を確かめた環境だけ。 */
export const TIMESTAMP_UNIT_NS: Readonly<Record<string, number>> = {
  "deno-intel-graphics-bmg-g21": 52.0833,
};

/** 表の行（無ければ undefined — 換算を推測しない）。 */
export const lookupTimestampUnitNs = (key: string | undefined): number | undefined =>
  key !== undefined && Object.hasOwn(TIMESTAMP_UNIT_NS, key) ? TIMESTAMP_UNIT_NS[key] : undefined;

/** 環境キー `key` の timestamp の 1 単位の ns（{@link TIMESTAMP_UNIT_NS}）。表に無ければ fail loudly。 */
export const timestampUnitNs = (key: string | undefined): number => {
  const unit = lookupTimestampUnitNs(key);
  if (unit === undefined) {
    throw new Error(
      `環境キー '${key}' の timestamp の 1 単位の ns が TIMESTAMP_UNIT_NS に無い` +
        "（推測で換算しない — docs/known-issues.md を見て行を足す）",
    );
  }
  return unit;
};
