/**
 * コードポイントの閉区間表（昇順・非重複）と二分探索。
 *
 * テキスト層は Unicode の分類を TS で再実装しない。分類の正本はエクスポータ側の
 * Python / Rust（`tokenizers`）で、そこが**全コードポイントを実評価して畳んだ**閉区間表を
 * ここが引くだけにする。判定の実装がここ 1 つなので、表さえ正しければ分類はずれない。
 */

/** 両端を含むコードポイント区間の昇順リスト。 */
export type CodeRanges = readonly (readonly [number, number])[];

/** cp が区間表に含まれるか。 */
export const inCodeRanges = (ranges: CodeRanges, cp: number): boolean => {
  let low = 0;
  let high = ranges.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const [start, end] = ranges[mid];
    if (cp < start) high = mid - 1;
    else if (cp > end) low = mid + 1;
    else return true;
  }
  return false;
};

/**
 * 外部境界（資産 JSON）の構造検査。壊れた表を黙って空表として使わない。
 *
 * MUST: **昇順・非重複**まで見る。{@link inCodeRanges} は二分探索なので、順序が崩れた表は
 * 例外にならず「静かに別の文字分類」になる（`\p{L}` の判定が 1 区間ぶん抜けるだけで
 * pre-token の切れ目が変わり、id 列が別物になる）。
 */
export const parseCodeRanges = (raw: unknown, label: string): CodeRanges => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 区間表が配列でない`);
  const ranges: (readonly [number, number])[] = [];
  for (const [index, entry] of raw.entries()) {
    if (
      !Array.isArray(entry) || entry.length !== 2 ||
      typeof entry[0] !== "number" || typeof entry[1] !== "number"
    ) {
      throw new Error(`${label}[${index}]: 区間が [start, end] の数値対でない`);
    }
    if (entry[0] > entry[1]) {
      throw new Error(`${label}[${index}]: 区間 [${entry[0]}, ${entry[1]}] の始端が終端より大きい`);
    }
    const previous = ranges[index - 1];
    if (previous !== undefined && previous[1] >= entry[0]) {
      throw new Error(
        `${label}[${index}]: 区間 [${entry[0]}, ${entry[1]}] が前の区間 ` +
          `[${previous[0]}, ${previous[1]}] と重なる / 昇順でない`,
      );
    }
    ranges.push([entry[0], entry[1]]);
  }
  return ranges;
};

/**
 * 文字列をコードポイント配列にする。
 *
 * 対になっていないサロゲートは**受け付けない** — Python / Rust の str には載らない値で
 * 正本の出力が定義されない。黙って U+FFFD に潰すと id 列が静かに変わるので落とす
 * （JS の文字列型でのみ表現できる入力なので、移植で最も忘れやすい防御）。
 */
export const toCodePoints = (text: string): number[] => {
  const out: number[] = [];
  for (const ch of text) {
    const cp = ch.codePointAt(0) as number;
    if (cp >= 0xd800 && cp <= 0xdfff) {
      throw new Error(`対にならないサロゲート U+${cp.toString(16).toUpperCase()} が入力にある`);
    }
    out.push(cp);
  }
  return out;
};

/** コードポイント 1 つの UTF-8 バイト長。 */
export const utf8Length = (cp: number): number =>
  cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
