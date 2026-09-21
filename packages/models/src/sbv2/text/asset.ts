/**
 * 実行時資産のトークナイザ JSON（`karume dist` が配る形）を検査して {@link DebertaTokenizer} に
 * するところまで。
 *
 * 見るのは資産の構造だけで、GPU も実行状態も触らない — 依存は `./tokenizer.ts` と資産 JSON の
 * 共通値域門（`../../text/asset-gates.ts`）に閉じる。
 *
 * MUST: `sbv2/pipeline.ts` を import しない。`text/` は pipeline の葉として片方向に保つ
 * （逆向きの依存を 1 本でも入れると、資産門だけを GPU 無しで叩けなくなる）。
 */

import { isRecord } from "../../config/readers.ts";
import { MAX_CODE_POINT } from "../../text/asset-gates.ts";
import { type CleanRanges, DebertaTokenizer } from "./tokenizer.ts";

/**
 * `cleanRanges` の区間表を検査して読む。
 *
 * MUST: 整数・コードポイント範囲・`start <= end`・**昇順かつ非重複**まで見る。`inRanges`
 * （`text/tokenizer.ts`）は二分探索なので、この前提が破れても例外は出ず**黙って外す** —
 * 除去 / 空白化の規則だけが変わった `bertText` から `inputIds` と `baseWord2ph` が同じ
 * 壊れ方で作られるため、`text/model-input.ts` の長さ突合門も通り、別の BERT 埋め込みで合成した
 * 音がそのまま出る。並べ替えて救わない（資産の不正として構築時に落とす）。
 */
const parseRanges = (raw: unknown, where: string): (readonly [number, number])[] => {
  if (!Array.isArray(raw)) throw new Error(`${where}: 区間表が配列でない`);
  let previousEnd = -1;
  return raw.map((entry, index) => {
    if (
      !Array.isArray(entry) || entry.length !== 2 ||
      typeof entry[0] !== "number" || typeof entry[1] !== "number"
    ) {
      throw new Error(`${where}[${index}]: 区間が [start, end] の数値対でない`);
    }
    const [start, end]: [number, number] = [entry[0], entry[1]];
    if (
      !Number.isInteger(start) || !Number.isInteger(end) ||
      start < 0 || end > MAX_CODE_POINT
    ) {
      throw new Error(
        `${where}[${index}]: 区間 [${start}, ${end}] が 0..${MAX_CODE_POINT} の整数対でない`,
      );
    }
    if (start > end) {
      throw new Error(`${where}[${index}]: 区間 [${start}, ${end}] の start が end より大きい`);
    }
    if (start <= previousEnd) {
      throw new Error(
        `${where}[${index}]: 区間 [${start}, ${end}] が直前の終端 ${previousEnd} 以下から始まる` +
          "（昇順・非重複でない — 二分探索が黙って外す）",
      );
    }
    previousEnd = end;
    return [start, end] as const;
  });
};

/** i32 の上限（トークン id は最終的に `Int32Array` へ書かれる）。 */
const MAX_TOKEN_ID = 2147483647;

/**
 * 実行時資産のトークナイザ JSON（`karume dist` が配る形）を検査して読む。
 *
 * MUST: 構造を検査してから使う。壊れた語彙表は「読めない」ではなく**全トークンが `[UNK]`**
 * という形で沈黙し、BERT 特徴だけが静かに無意味になる（`text/tokenizer.ts` の doc）。
 */
export const parseTokenizerAsset = (raw: unknown, where: string): DebertaTokenizer => {
  if (!isRecord(raw)) throw new Error(`${where}: オブジェクトでない`);
  const special = raw["special"];
  if (!isRecord(special)) throw new Error(`${where}.special: オブジェクトでない`);
  const [clsId, sepId, unkId] = [special["clsId"], special["sepId"], special["unkId"]];
  if (typeof clsId !== "number" || typeof sepId !== "number" || typeof unkId !== "number") {
    throw new Error(`${where}.special: clsId / sepId / unkId が数値でない`);
  }
  // MUST: 整数かつ i32 の範囲。id は `Int32Array` へ書かれるので、非整数は**黙って切り捨て
  // られ**、範囲外は wrap する — どちらも「別のトークンを指す」沈黙誤値になる（語彙の行数に
  // 収まることは `fromVocabText` が語彙表を持つ側で見る）。
  for (const [name, id] of [["clsId", clsId], ["sepId", sepId], ["unkId", unkId]] as const) {
    if (!Number.isInteger(id) || id < 0 || id > MAX_TOKEN_ID) {
      throw new Error(`${where}.special.${name}: 0..${MAX_TOKEN_ID} の整数でない（${id}）`);
    }
  }
  const vocabText = raw["vocabText"];
  if (typeof vocabText !== "string" || vocabText.length === 0) {
    throw new Error(`${where}.vocabText: 空`);
  }
  const clean = raw["cleanRanges"];
  if (!isRecord(clean)) throw new Error(`${where}.cleanRanges: オブジェクトでない`);
  const ranges: CleanRanges = {
    removed: parseRanges(clean["removed"], `${where}.cleanRanges.removed`),
    spaced: parseRanges(clean["spaced"], `${where}.cleanRanges.spaced`),
  };
  return DebertaTokenizer.fromVocabText(vocabText, ranges, { clsId, sepId, unkId });
};
