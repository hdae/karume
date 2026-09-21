/**
 * Anima の資産 JSON が載せる閉区間表の読み取り（表そのものと二分探索は
 * `src/text/code-ranges.ts`）。
 *
 * ここが持つのは**外部境界の検査**と、正規化が要る UTF-8 長だけ。検査の文言は sbv2 の同型
 * `parseRanges` と揃っていないので、1 本にまとめず family ごとに置く。
 */

import { assertCodePoint } from "../../text/asset-gates.ts";
import type { CodeRanges } from "../../text/code-ranges.ts";

/**
 * 外部境界（資産 JSON）の構造検査。壊れた表を黙って空表として使わない。
 *
 * MUST: **整数・コードポイント範囲・昇順・非重複**まで見る。`inCodeRanges` は二分探索
 * なので、この前提が破れても例外にならず「静かに別の文字分類」になる（`[65.5, 90]` は境界だけ
 * を半端にずらし、順序が崩れた表は `\p{L}` の判定が 1 区間ぶん抜ける — どちらも pre-token の
 * 切れ目が変わって id 列が別物になる）。値域の規律は sbv2 の同型 `parseRanges` と同じ。
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
    assertCodePoint(entry[0], `${label}[${index}]`);
    assertCodePoint(entry[1], `${label}[${index}]`);
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

/** コードポイント 1 つの UTF-8 バイト長。 */
export const utf8Length = (cp: number): number =>
  cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
