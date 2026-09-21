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
