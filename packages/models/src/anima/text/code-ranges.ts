/**
 * Anima の正規化が要る UTF-8 長（閉区間表そのもの・二分探索・資産 JSON からの読み取りは
 * `src/text/code-ranges.ts`）。
 */

/** コードポイント 1 つの UTF-8 バイト長。 */
export const utf8Length = (cp: number): number =>
  cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
