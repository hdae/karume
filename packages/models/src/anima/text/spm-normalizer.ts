/**
 * T5（SentencePiece）の `Precompiled` 正規化器を、焼いた表を引くだけで再現する。
 *
 * 正本は Rust の `tokenizers::normalizers::Precompiled`。その実挙動は SentencePiece の
 * 正規化とは別物で、**書記素クラスタ単位の丸ごと置換 + 最短接頭辞勝ち**という形をしている:
 *
 *     クラスタ g ごとに:
 *       UTF-8 長(g) < 6 かつ g のいずれかの接頭辞が規則表にある
 *         → **最短**の一致接頭辞の値を出し、g の残りは捨てる
 *       さもなくば 1 コードポイントずつ引く（規則が無ければその文字のまま）
 *
 * 実測:
 *   `A`+U+0301+U+0301 → `Á`（3 文字目が消える）/ `A`+U+0302+U+0301 → `Â`
 *   （3cp 規則 `Ấ` ではなく 2cp 接頭辞が勝つ）。素朴に「最長一致」や NFKC で実装すると、
 *   この 2 例で確実に外れる。
 *
 * **DARTS トライを TS で復号しない / UAX#29 を TS で実装しない**。エクスポータが正本を
 * 全コードポイントに当てて、
 *   ・発火しうる規則だけ（真の接頭辞に規則を持たない・UTF-8 6 バイト未満）
 *   ・クラスタ境界に要る 3 つの閉区間表（extend / breakAfter / prepend）
 * に畳んだものをここが引く。畳み込みの同値は emit のたびに網羅 + 乱択で検査される。
 */

import { toCodePoints } from "../../text/code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges, utf8Length } from "./code-ranges.ts";

/** 丸ごと置換を試みるクラスタの UTF-8 長上限（正本 spm_precompiled の `len() < 6`）。 */
const CLUSTER_BYTE_LIMIT = 6;

const CR = 0x0d;
const LF = 0x0a;

/** 2 コードポイント鍵の合成キー（コードポイントは 0x10FFFF 以下なので衝突しない）。 */
const pairKey = (first: number, second: number): number => first * 0x110000 + second;

/** 焼いた正規化表。 */
export type SpmTables = {
  /** 1 コードポイント鍵の写像。 */
  readonly single: ReadonlyMap<number, string>;
  /** 2 コードポイント鍵の写像（真の接頭辞に規則を持たないものだけ）。 */
  readonly multi: ReadonlyMap<number, string>;
  /** 直前の文字と同じクラスタに合流する文字。 */
  readonly extend: CodeRanges;
  /** この文字の直後で必ずクラスタが切れる文字。 */
  readonly breakAfter: CodeRanges;
  /** 後続をクラスタに引き込む文字。 */
  readonly prepend: CodeRanges;
};

/** 焼いた表だけで `Precompiled` を再現する。 */
export const normalizeSpm = (tables: SpmTables, text: string): string => {
  const cps = toCodePoints(text);
  const out: string[] = [];
  let i = 0;
  while (i < cps.length) {
    const end = clusterEnd(tables, cps, i);
    let bytes = 0;
    for (let k = i; k < end; k++) bytes += utf8Length(cps[k]);
    let replacement: string | undefined;
    // MUST: 6 バイト境界を落とすと絵文字 / ハングル周りが静かに壊れる（丸ごと置換の経路に
    // 入らないはずのクラスタが入る）。
    if (bytes < CLUSTER_BYTE_LIMIT) {
      // 最短の一致接頭辞が勝つ — 1cp を先に引く。
      replacement = tables.single.get(cps[i]);
      if (replacement === undefined && end - i >= 2) {
        replacement = tables.multi.get(pairKey(cps[i], cps[i + 1]));
      }
    }
    if (replacement !== undefined) {
      out.push(replacement);
    } else {
      for (let k = i; k < end; k++) {
        out.push(tables.single.get(cps[k]) ?? String.fromCodePoint(cps[k]));
      }
    }
    i = end;
  }
  return out.join("");
};

/** i から始まるクラスタの終端（排他）。prepend* + 基底 + extend* / CRLF / 制御単独。 */
const clusterEnd = (tables: SpmTables, cps: readonly number[], i: number): number => {
  let j = i;
  while (j < cps.length && inCodeRanges(tables.prepend, cps[j])) j++;
  // prepend の直後が制御文字ならそこで切れる（引き込まない）。
  if (j < cps.length && !(j > i && inCodeRanges(tables.breakAfter, cps[j]))) {
    if (cps[j] === CR && cps[j + 1] === LF) {
      j += 2; // GB3: CR × LF は 1 クラスタ
    } else if (inCodeRanges(tables.breakAfter, cps[j])) {
      j += 1;
    } else {
      j += 1;
      while (j < cps.length && inCodeRanges(tables.extend, cps[j])) j++;
    }
  }
  return j;
};

/** 資産 / フィクスチャの JSON 形（Python 側 `SpmTables.to_json` と対）。 */
export const parseSpmTables = (raw: unknown, label: string): SpmTables => {
  if (typeof raw !== "object" || raw === null) throw new Error(`${label}: オブジェクトでない`);
  const obj = raw as Record<string, unknown>;
  const single = new Map<number, string>();
  for (const entry of asArray(obj["single"], `${label}.single`)) {
    if (!Array.isArray(entry) || typeof entry[0] !== "number" || typeof entry[1] !== "string") {
      throw new Error(`${label}.single: [cp, 文字列] でない`);
    }
    single.set(entry[0], entry[1]);
  }
  const multi = new Map<number, string>();
  for (const entry of asArray(obj["multi"], `${label}.multi`)) {
    if (
      !Array.isArray(entry) || typeof entry[0] !== "number" ||
      typeof entry[1] !== "number" || typeof entry[2] !== "string"
    ) {
      throw new Error(`${label}.multi: [cp, cp, 文字列] でない`);
    }
    multi.set(pairKey(entry[0], entry[1]), entry[2]);
  }
  return {
    single,
    multi,
    extend: parseCodeRanges(obj["extend"], `${label}.extend`),
    breakAfter: parseCodeRanges(obj["breakAfter"], `${label}.breakAfter`),
    prepend: parseCodeRanges(obj["prepend"], `${label}.prepend`),
  };
};

const asArray = (value: unknown, label: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${label}: 配列でない`);
  return value;
};
